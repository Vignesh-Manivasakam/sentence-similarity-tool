import os
import json
import logging
import datetime
import uuid
import filelock
import sqlite3
from typing import Any

from am_ais_assist.config import CACHE_DIR

logger = logging.getLogger(__name__)

# Separator for query-match pair text representation
_PAIR_SEPARATOR = " ||| "

def get_user_skills_file_path(user_id: str) -> str:
    """Get the persistent file path for a user's skills JSON profile (used for legacy lookup/fallbacks)."""
    # Ensure cache directory exists
    skills_dir = os.path.join(CACHE_DIR, "user_skills")
    os.makedirs(skills_dir, exist_ok=True)
    # Sanitize user_id to prevent directory traversal
    safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ("_", "-"))
    if not safe_user_id:
        safe_user_id = "default_user"
    return os.path.join(skills_dir, f"user_{safe_user_id}_skills.json")

def get_db_connection():
    """Return a connection to the central SQLite database for user skills, creating tables if needed."""
    db_path = os.path.join(CACHE_DIR, "user_skills.db")
    conn = sqlite3.connect(db_path, timeout=10)
    # Enable WAL mode for high-concurrency read/write operations
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.Error:
        pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_skills (
            user_id TEXT PRIMARY KEY,
            profile_json TEXT,
            last_updated TEXT,
            version INTEGER
        )
    """)
    conn.commit()
    return conn

def load_user_skills(user_id: str) -> dict:
    """Load the user's skill profile from the SQLite database, migrating legacy JSON if needed."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT profile_json FROM user_skills WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            conn.close()
            return json.loads(row[0])
    except Exception as e:
        logger.error("Failed to load skills from DB for user %s: %s", user_id, e)

    # Fallback to legacy JSON file
    file_path = get_user_skills_file_path(user_id)
    lock_path = file_path + ".lock"
    
    profile = {
        "user_id": user_id,
        "version": 0,
        "last_updated": "",
        "skills": [],
        "history": []
    }
    
    if os.path.exists(file_path):
        try:
            with filelock.FileLock(lock_path, timeout=5):
                with open(file_path, "r", encoding="utf-8") as f:
                    profile = json.load(f)
            # Auto-migrate to SQLite database
            save_user_skills(user_id, profile)
            logger.info("Migrated user %s skills from JSON file to central SQLite database.", user_id)
        except Exception as e:
            logger.error("Failed to read/migrate legacy skills file for user %s: %s", user_id, e)
            
    return profile

def save_user_skills(user_id: str, profile: dict) -> None:
    """Save the user's skill profile to the central SQLite database with a fallback to legacy JSON file."""
    try:
        conn = get_db_connection()
        version = profile.get("version", 0)
        last_updated = profile.get("last_updated", "")
        profile_json = json.dumps(profile, indent=2)
        
        with conn:
            conn.execute("""
                INSERT INTO user_skills (user_id, profile_json, last_updated, version)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_json=excluded.profile_json,
                    last_updated=excluded.last_updated,
                    version=excluded.version
            """, (user_id, profile_json, last_updated, version))
        conn.close()
    except Exception as e:
        logger.error("Failed to save skills for user %s to database: %s", user_id, e)
        
        # Fallback to local JSON file
        file_path = get_user_skills_file_path(user_id)
        lock_path = file_path + ".lock"
        try:
            with filelock.FileLock(lock_path, timeout=5):
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(profile, f, indent=2)
            logger.warning("Fell back to saving JSON file for user %s due to DB failure.", user_id)
        except Exception as fe:
            logger.error("Critical: Fallback save failed for user %s: %s", user_id, fe)

def get_user_skills_prompt_context(user_id: str) -> str:
    """Load user skills and format active ones into instructions for system prompt injection.
    
    Limits skills to top 10 and examples per skill to 3 to manage token costs.
    """
    profile = load_user_skills(user_id)
    skills = profile.get("skills", [])
    
    # Active threshold is confidence_score >= 0.70
    active_skills = [s for s in skills if float(s.get("confidence_score", 0.0)) >= 0.70]
    if not active_skills:
        return ""
        
    # Cap to top 10 most confident/frequent skills
    active_skills = sorted(active_skills, key=lambda s: float(s.get("confidence_score", 0.0)), reverse=True)[:10]
    
    lines = [
        "",
        "🔑 User-Specific Matching Rules",
        "Based on your historical matching decisions, apply the following customized rules:",
    ]
    
    for i, skill in enumerate(active_skills, 1):
        desc = skill.get("pattern_description", "")
        lines.append(f"{i}. Rule: {desc}")
        examples = skill.get("examples", [])[:3]  # Limit examples to 3 to prevent bloat
        if examples:
            lines.append("   Examples:")
            for ex in examples:
                base = ex.get("base_text", "")
                new = ex.get("new_text", "")
                decision = ex.get("user_decision", "")
                remark = ex.get("user_remark", "")
                ex_line = f"     * Base: \"{base}\" | New: \"{new}\" -> Decision: {decision}"
                if remark:
                    ex_line += f" (Reason: {remark})"
                lines.append(ex_line)
                
    return "\n".join(lines)

def extract_and_update_skills(user_id: str, new_reviews: list[dict]) -> None:
    """Send reviews to the LLM to extract new skills and merge them into the user's profile.
    
    Runs in a background thread or process.
    """
    if not new_reviews:
        return
        
    from am_ais_assist.llm_service import client, LLM_MODEL, _extract_json  # avoid circular imports
    
    if client is None:
        logger.warning("LLM client not initialized — skipping skill extraction.")
        return

    # Load current profile
    profile = load_user_skills(user_id)
    
    # Format reviews for user prompt
    reviews_formatted = []
    for r in new_reviews:
        reviews_formatted.append({
            "base_text": r.get("base_text", ""),
            "new_text": r.get("new_text", ""),
            "ai_original_decision": f"{r.get('ai_level', '')} ({r.get('ai_score', '')})",
            "user_verdict": r.get("verdict", ""),
            "user_remark": r.get("user_remark", "")
        })
        
    system_prompt = """You are a machine learning prompt engineering and requirements analysis expert.
Your job is to maintain a persistent profile of user-specific matching rules (skills) based on human-in-the-loop corrections.

A skills profile is a JSON object containing:
- "user_id": The user's ID
- "version": The current profile version (increment by 1 on update)
- "last_updated": Current UTC timestamp (ISO format)
- "skills": An array of skill objects:
  - "id": A unique string identifier (e.g. UUID or prefix)
  - "pattern_description": A clear instruction explaining the preference (e.g., "Treat pressure units bar and Pa as equivalent after conversion")
  - "rules": A list of short text rules (e.g. ["1 bar = 100000 Pa"])
  - "examples": A list of examples (max 3 per skill) where:
    - "base_text": The base requirement text
    - "new_text": The new requirement text
    - "user_decision": The user's final decision ("Equivalent", "Related", "Contradictory")
    - "user_remark": The user's explanation
  - "confidence_score": A float between 0.0 and 1.0 (new patterns start at 0.50; reinforce when user makes consistent choices, decrease when conflicting)
  - "occurrence_count": Number of times this pattern has occurred
- "history": An audit trail listing version, timestamp, and change_summary.

You must:
1. Group corrections by semantic patterns or business rules.
2. Deduplicate: If a new correction matches an existing skill, add it to the examples (keep max 3), increment occurrence_count, and increase confidence_score by 0.15 (cap at 0.99).
3. Conflict Resolution: If new corrections contradict an existing skill (e.g. user previously said X is equivalent, but Y is related for a similar case), refine the pattern_description to handle the exception, or reduce the confidence_score by 0.20 if they are inconsistent.
4. Create new skills: If a correction represents a new pattern, create a new skill object with confidence_score = 0.50 and occurrence_count = 1.
5. Cap skills: Keep only the most relevant/confident skills (active ones are confidence >= 0.70). Max 10 skills in the list.
6. Update history and increment version.

Ensure your output is strictly valid JSON matching the schema of the existing skills profile. Do not return markdown code blocks, formatting, or extra text outside of the JSON."""

    user_message = f"""Existing Skills Profile:
{json.dumps(profile, indent=2)}

New User Corrections and Remarks:
{json.dumps(reviews_formatted, indent=2)}"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.0,  # deterministic skill updates
            n=1,
        )
        
        content = response.choices[0].message.content or ""
        json_str = _extract_json(content)
        updated_profile = json.loads(json_str)
        
        # Verify basic keys
        if "skills" in updated_profile and "user_id" in updated_profile:
            save_user_skills(user_id, updated_profile)
            logger.info("Successfully updated skill profile for user %s to version %s", user_id, updated_profile.get("version", 0))
        else:
            logger.error("LLM did not return a valid user skills profile schema.")
            
    except Exception as exc:
        logger.error("Failed to run skill extraction LLM: %s", exc)
