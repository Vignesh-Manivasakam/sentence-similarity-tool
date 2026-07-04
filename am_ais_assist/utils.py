from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import TruncatedSVD  # P-1: TruncatedSVD scales better than PCA

from am_ais_assist.preprocess import preprocess_data

# P-1: cap the number of points fed into dimensionality reduction to avoid memory cliff.
# PCA on (N+M) × 1536 is O(n × d²) — 5000 points → ~7 billion ops.
_MAX_PLOT_SAMPLES = 2000


def read_excel_dataframe(file_obj: Any) -> pd.DataFrame | None:
    """Read an Excel file object into a DataFrame without consuming the stream.

    Seeks to 0 before reading and seeks back to 0 after reading so the caller
    can still use the file object for other purposes (e.g. hashing or
    ``excel_to_json``).

    Args:
        file_obj: A file-like object (must support ``seek`` and ``read``).
                  Typically a Streamlit ``UploadedFile`` instance.

    Returns:
        A ``pd.DataFrame`` containing all columns and rows from the Excel
        file, or ``None`` if reading fails for any reason.
    """
    try:
        file_obj.seek(0)
        df = pd.read_excel(file_obj)
        file_obj.seek(0)
        logging.info(
            "read_excel_dataframe: read %d rows, %d columns", len(df), len(df.columns)
        )
        return df
    except Exception as exc:  # noqa: BLE001
        logging.warning("read_excel_dataframe failed: %s", exc)
        try:
            file_obj.seek(0)
        except Exception:  # noqa: BLE001
            pass
        return None


def excel_to_json(
    file_obj: Any,
    tokenizer: Any,
    id_col: str,
    text_col: str,
    meta_cols: list[str] | None = None,
) -> tuple[list[dict], int]:
    """Read an Excel file and return preprocessed requirement records.

    Args:
        file_obj:   File-like object pointing to an ``.xlsx`` / ``.xls`` file.
        tokenizer:  Tokenizer used for truncation (may be ``None``).
        id_col:     Name of the column used as the unique row identifier.
        text_col:   Name of the column containing the requirement text.
        meta_cols:  Optional list of additional columns to carry through.

    Returns:
        Tuple of ``(processed_data, skipped_count)`` where ``processed_data``
        is a list of dicts with keys ``Object_Identifier``, ``Object_Text``,
        ``Cleaned_Text``, ``Hierarchy``, ``Truncated`` (plus any meta cols),
        and ``skipped_count`` is the number of rows dropped during preprocessing.

    Raises:
        ValueError: If the file is too large, has an unsupported extension,
                    or the requested columns are not present.
    """
    try:
        if meta_cols is None:
            meta_cols = []

        # M-7: Validate file size before reading
        file_obj.seek(0, 2)
        file_size = file_obj.tell()
        file_obj.seek(0)
        from am_ais_assist.config import MAX_FILE_SIZE

        if file_size > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {file_size / 1024 / 1024:.1f} MB "
                f"(limit: {MAX_FILE_SIZE // (1024 * 1024)} MB)"
            )

        # M-7: Validate file extension
        _ALLOWED_EXTENSIONS = {".xlsx", ".xls"}
        name = getattr(file_obj, "name", "")
        if name and not any(name.lower().endswith(ext) for ext in _ALLOWED_EXTENSIONS):
            raise ValueError(
                f"Unsupported file type '{name}'. Allowed: {sorted(_ALLOWED_EXTENSIONS)}"
            )

        file_obj.seek(0)
        df = pd.read_excel(file_obj, dtype={text_col: str, id_col: str})

        all_cols = [id_col, text_col] + meta_cols
        if not all(c in df.columns for c in all_cols):
            missing_cols = [c for c in all_cols if c not in df.columns]
            raise ValueError(f"Selected columns {missing_cols} not found in Excel file")

        df[text_col] = df[text_col].fillna("").astype(str)
        df[id_col] = df[id_col].fillna("").astype(str)

        # Additional validation for text column
        if df[text_col].str.strip().eq("").all():
            raise ValueError("Text column contains only empty or whitespace values")

        logging.info("Excel file processed: %s rows", len(df))
        logging.debug("Excel columns: %s", df.columns.tolist())
        logging.debug("Sample data: %s", df.head(2).to_dict())

        # Select and rename columns
        df_subset = df[[id_col, text_col] + meta_cols].copy()
        rename_map = {id_col: "Object_Identifier", text_col: "Object_Text"}
        df_subset.rename(columns=rename_map, inplace=True)

        # Convert to dictionary records
        data = df_subset.to_dict(orient="records")

        processed_data, skipped_empty_count = preprocess_data(data, tokenizer)

        logging.info(
            "Preprocessed %s entries, skipped %s empty texts",
            len(processed_data),
            skipped_empty_count,
        )
        return processed_data, skipped_empty_count

    except Exception as e:
        logging.error("Failed to process Excel file: %s", e)
        raise ValueError(f"Error processing Excel file: {str(e)}") from e


def plot_embeddings(
    base_embeddings: np.ndarray,
    user_embeddings: np.ndarray,
    base_data: list[dict],
    user_data: list[dict],
) -> Any:
    """Generate a 3-D embedding visualisation with smart duplicate handling.

    Args:
        base_embeddings: NumPy array of shape ``(N, D)`` for the base file.
        user_embeddings: NumPy array of shape ``(M, D)`` for the query file.
        base_data:       Processed entry dicts for the base file.
        user_data:       Processed entry dicts for the query file.

    Returns:
        A Plotly ``Figure`` object ready for ``st.plotly_chart()``.
    """
    try:
        # Filter out zero embeddings (exact matches) from user embeddings
        non_zero_indices = []
        filtered_user_embeddings = []
        filtered_user_data = []

        for i, embedding in enumerate(user_embeddings):
            # Check if embedding is not all zeros (indicating it was actually computed)
            if not np.allclose(embedding, 0):
                non_zero_indices.append(i)
                filtered_user_embeddings.append(embedding)
                filtered_user_data.append(user_data[i])

        # Convert to numpy array if we have any non-zero embeddings
        if filtered_user_embeddings:
            filtered_user_embeddings = np.array(filtered_user_embeddings)
        else:
            filtered_user_embeddings = np.array([]).reshape(0, base_embeddings.shape[1])

        # Check if we should show only base embeddings
        show_only_base = len(filtered_user_embeddings) == 0

        if show_only_base:
            # Only show base embeddings when all queries are exact matches
            logging.info("All queries are exact matches - showing only base embeddings")
            combined_embeddings = base_embeddings
            categories = ["Base"] * len(base_embeddings)
            texts = [entry["Original_Text"] for entry in base_data]

            # Add note about exact matches
            title = "3D Embedding Visualization - Base Only (All queries are exact matches)"
        else:
            # Combine base and filtered user embeddings
            combined_embeddings = np.vstack((base_embeddings, filtered_user_embeddings))
            base_count = len(base_embeddings)
            user_count = len(filtered_user_embeddings)
            categories = ["Base"] * base_count + ["Query"] * user_count
            texts = [entry["Original_Text"] for entry in base_data] + [
                entry["Original_Text"] for entry in filtered_user_data
            ]

            title = (
                f"3D Embedding Visualization ({len(filtered_user_embeddings)} unique queries shown)"
            )

        # P-1: sub-sample before dimensionality reduction to avoid memory cliff
        if len(combined_embeddings) > _MAX_PLOT_SAMPLES:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(combined_embeddings), _MAX_PLOT_SAMPLES, replace=False)
            combined_embeddings = combined_embeddings[idx]
            categories = [categories[i] for i in idx]
            texts = [texts[i] for i in idx]

        # P-1: TruncatedSVD is O(n*d*k) — much faster than PCA's O(n*d²) for high-d embeddings
        svd = TruncatedSVD(n_components=3, random_state=42)
        reduced = svd.fit_transform(combined_embeddings)

        # Color mapping based on whether we're showing both types
        if show_only_base:
            color_map = {"Base": "#00d4ff"}
        else:
            color_map = {"Base": "#00d4ff", "Query": "#ffaa00"}

        fig = px.scatter_3d(
            x=reduced[:, 0],
            y=reduced[:, 1],
            z=reduced[:, 2],
            color=categories,
            color_discrete_map=color_map,
            hover_data={"text": texts, "category": categories},
            title=title,
            labels={"x": "Main Pattern", "y": "Secondary Pattern", "z": "Tertiary Pattern"},
            opacity=0.8,
            size=[10] * len(reduced),
            size_max=15,
        )

        fig.update_traces(
            hovertemplate="%{customdata[0]}<extra></extra>", marker=dict(line=dict(width=0))
        )

        fig.update_layout(
            scene=dict(
                xaxis_title="Main Pattern",
                yaxis_title="Secondary Pattern",
                zaxis_title="Tertiary Pattern",
                bgcolor="#1e1e2f",
            ),
            paper_bgcolor="#1e1e2f",
            font=dict(color="white"),
            clickmode="event+select",
            legend=dict(title="Embedding Type", font=dict(color="white")),
            width=1000,
            height=640,
            margin=dict(l=20, r=20, t=50, b=20),
        )

        return fig

    except Exception as e:
        logging.error("Error creating embedding plot: %s", e)
        fig = px.scatter_3d(title="Embedding Visualization (Error occurred)")
        return fig
