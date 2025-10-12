import pandas as pd
import streamlit as st
import plotly.express as px
from openpyxl import Workbook
from openpyxl.cell.text import InlineFont
from openpyxl.cell.rich_text import CellRichText, TextBlock
import io
import logging
import difflib
import html
import re


def df_to_html_table(df, base_data, user_data):
    """Generates a styled HTML table from the results DataFrame with truncation indicators and colored LLM relationships."""
    html_str = '<table class="results-table">'
    
    all_columns = df.columns.tolist()
    
    query_cols = sorted([c for c in all_columns if c.startswith('Query_') and c not in ['Query_Sentence_Highlighted', 'Query_Sentence', 'Query_Sentence_Cleaned_text', 'Query_Object_Identifier']])
    matched_cols = sorted([c for c in all_columns if c.startswith('Matched_') and c not in ['Matched_Sentence_Highlighted', 'Matched_Sentence', 'Matched_Sentence_Cleaned_text', 'Matched_Object_Identifier']])
    
   
    # score_cols = [c for c in all_columns if 'Similarity_' in c]
    
    llm_cols = [c for c in all_columns if c in ['Similarity_Score', 'Similarity_Level', 'Remark'] or 'LLM_' in c]

    # Updated visible_columns 
    visible_columns = (
        ['Query_Object_Identifier', 'Query_Sentence_Highlighted'] +
        query_cols +
        ['Matched_Object_Identifier', 'Matched_Sentence_Highlighted'] +
        matched_cols +
        llm_cols  
    )
    visible_columns = [col for col in visible_columns if col in df.columns]

    html_str += '<tr style="background-color: #444; color: white; text-align: left;">'
    for col in visible_columns:
        header_text = col.replace("_Highlighted", "").replace("_", " ").title()
        html_str += f'<th class="sentence-column" style="width: auto;">{header_text}</th>'
    html_str += '</tr>'

    for idx, row in df.iterrows():
        html_str += '<tr>'
        for col in visible_columns:
            text = str(row.get(col, 'N/A'))
            if col == 'Query_Sentence_Highlighted':
                query_id = row['Query_Object_Identifier']
                is_truncated = next((entry['Truncated'] for entry in user_data if entry['Object_Identifier'] == query_id), False)
                if not is_valid_html(text):
                    text = html.escape(text)
                text = f"<span style='color:yellow'>⚠️</span> {text}" if is_truncated else text
                html_str += f'<td class="sentence-column">{text}</td>'
            elif col == 'Matched_Sentence_Highlighted':
                match_id = row['Matched_Object_Identifier']
                is_truncated = next((entry['Truncated'] for entry in base_data if entry['Object_Identifier'] == match_id), False)
                if not is_valid_html(text):
                    text = html.escape(text)
                text = f"<span style='color:yellow'>⚠️</span> {text}" if is_truncated else text
                html_str += f'<td class="sentence-column">{text}</td>'
            elif col == 'Similarity_Level':
                header_text = 'Similarity Level'
                # Enhanced color coding for LLM relationships
                color = '#4CAF50' if any(word in text.lower() for word in ['equivalent', 'exact', 'match']) else \
                        '#ff6b6b' if any(word in text.lower() for word in ['contradictory', 'opposite', 'different']) else \
                        '#FFA500' if any(word in text.lower() for word in ['related', 'similar', 'partial']) else \
                        '#9E9E9E' if any(word in text.lower() for word in ['threshold', 'error', 'n/a']) else 'inherit'
                html_str += f'<td style="color: {color}; font-weight: bold;">{html.escape(text)}</td>'
            elif col == 'Similarity Score':
                header_text = 'Similarity Score'
                # Format LLM Score with appropriate styling
                if text != 'N/A' and text != 'Error':
                    try:
                        score_val = float(text)
                        color = '#4CAF50' if score_val >= 0.8 else '#FFA500' if score_val >= 0.5 else '#ff6b6b'
                        html_str += f'<td style="color: {color}; font-weight: bold;">{html.escape(text)}</td>'
                    except (ValueError, TypeError):
                        html_str += f'<td style="color: #9E9E9E;">{html.escape(text)}</td>'
                else:
                    html_str += f'<td style="color: #9E9E9E;">{html.escape(text)}</td>'
            else:
                html_str += f'<td>{html.escape(text)}</td>'
        html_str += '</tr>'
    
    html_str += '</table>'
    return html_str if is_valid_html(html_str) else html.escape(html_str)


def is_valid_html(text):
    try:
        invalid_tag_pattern = r'<[\w\s]*[<>][\w\s]*>'
        return not bool(re.search(invalid_tag_pattern, text))
    except:
        return False


def highlight_word_differences(query_text, matched_text):
    try:
        query_words = query_text.split()
        matched_words = matched_text.split()
        diff = list(difflib.ndiff(query_words, matched_words))
        highlighted_query, highlighted_matched = [], []

        for d in diff:
            if d.startswith('  '):
                word = html.escape(d[2:])
                highlighted_query.append(word)
                highlighted_matched.append(word)
            elif d.startswith('- '):
                word = html.escape(d[2:])
                highlighted_query.append(f'<span style="color:#ff6b6b; font-weight:bold">{word}</span>')
            elif d.startswith('+ '):
                word = html.escape(d[2:])
                highlighted_matched.append(f'<span style="color:#00d4ff; font-weight:bold">{word}</span>')

        return ' '.join(highlighted_query), ' '.join(highlighted_matched)
    except:
        return html.escape(str(query_text)), html.escape(str(matched_text))


def create_highlighted_excel(df, base_data, user_data):
    """Generate an Excel file with red/blue diff highlighting and yellow ⚠️ for truncated text."""
    try:
        wb = Workbook()
        ws = wb.active
        ws.title = "LLM Analysis Results"  # Updated title

        all_cols = df.columns.tolist()
        non_meta_query = ['Query_Object_Identifier', 'Query_Sentence', 'Query_Sentence_Cleaned_text', 'Query_Sentence_Highlighted']
        non_meta_matched = ['Matched_Object_Identifier', 'Matched_Sentence', 'Matched_Sentence_Cleaned_text', 'Matched_Sentence_Highlighted']

        query_meta_cols = sorted([c for c in all_cols if c.startswith('Query_') and c not in non_meta_query])
        matched_meta_cols = sorted([c for c in all_cols if c.startswith('Matched_') and c not in non_meta_matched])

        # UPDATED: Removed similarity columns, only include LLM columns
        final_headers_ordered = (
            ['Query_Object_Identifier', 'Query_Sentence'] +
            query_meta_cols +
            ['Matched_Object_Identifier', 'Matched_Sentence'] +
            matched_meta_cols +
            ['Similarity_Score', 'Similarity_Level', 'Remark']  # Renamed columns
        )
        headers_to_write = [h for h in final_headers_ordered if h in all_cols]
        ws.append([h.replace("_", " ").title() for h in headers_to_write])

        df_to_write = df[headers_to_write]

        for _, row in df_to_write.iterrows():
            row_idx = ws.max_row + 1
            for col_idx, col_name in enumerate(headers_to_write, start=1):

                if col_name == 'Query_Sentence' or col_name == 'Matched_Sentence':
                    q_text = str(row['Query_Sentence'])
                    m_text = str(row['Matched_Sentence'])
                    q_id = row['Query_Object_Identifier']
                    m_id = row['Matched_Object_Identifier']

                    q_words = q_text.split()
                    m_words = m_text.split()
                    diff = list(difflib.ndiff(q_words, m_words))

                    blocks = []
                    if col_name == 'Query_Sentence':
                        is_truncated = next((e['Truncated'] for e in user_data if e['Object_Identifier'] == q_id), False)
                        if is_truncated:
                            blocks.append(TextBlock(InlineFont(color="FFFF00", b=True), "⚠️ "))
                        for d in diff:
                            if d.startswith('- '):
                                blocks.append(TextBlock(InlineFont(color="FF0000", b=True), d[2:] + " "))
                            elif d.startswith('  '):
                                blocks.append(TextBlock(InlineFont(color="000000"), d[2:] + " "))
                    elif col_name == 'Matched_Sentence':
                        is_truncated = next((e['Truncated'] for e in base_data if e['Object_Identifier'] == m_id), False)
                        if is_truncated:
                            blocks.append(TextBlock(InlineFont(color="FFFF00", b=True), "⚠️ "))
                        for d in diff:
                            if d.startswith('+ '):
                                blocks.append(TextBlock(InlineFont(color="0000FF", b=True), d[2:] + " "))
                            elif d.startswith('  '):
                                blocks.append(TextBlock(InlineFont(color="000000"), d[2:] + " "))

                    if blocks:
                        ws.cell(row=row_idx, column=col_idx).value = CellRichText(blocks)
                    else:
                        ws.cell(row=row_idx, column=col_idx, value=row[col_name])

                else:
                    ws.cell(row=row_idx, column=col_idx, value=str(row[col_name]) if pd.notna(row[col_name]) else 'N/A')

        for col in ws.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col:
                try:
                    if cell.value and len(str(cell.value)) > max_len:
                        max_len = len(str(cell.value))
                except:
                    pass
            ws.column_dimensions[col_letter].width = min(max_len + 2, 50)

        output = io.BytesIO()
        wb.save(output)
        output.seek(0)
        return output.getvalue()

    except Exception as e:
        logging.error(f"Error creating Excel file: {str(e)}")
        raise


def display_summary(results):
    """Updated to display LLM-based summary instead of similarity levels"""
    try:
        df = pd.DataFrame(results)
        if df.empty:
            st.warning("No results to display summary for.")
            return
        
        num_queries = len(df['Query_Object_Identifier'].unique())
        
        # UPDATED: Use LLM columns instead of similarity columns
        if 'Similarity_Score' in df.columns:
            # Calculate LLM score statistics (excluding N/A and Error values)
            numeric_llm_scores = pd.to_numeric(df['Similarity_Score'], errors='coerce').dropna()
            avg_llm_score = numeric_llm_scores.mean() if len(numeric_llm_scores) > 0 else 0
            
            # Get LLM relationship distribution
            llm_relationship_counts = df['Similarity_Level'].value_counts().to_dict()
            
            st.markdown('<div class="summary-card">', unsafe_allow_html=True)
            st.markdown("**Analysis Summary**")  # Updated title
            st.write(f"- **Queries Processed**: {num_queries}")
            st.write(f"- **Average Similarity Score**: {avg_llm_score:.4f}")
            st.write(f"- **Total LLM Analyses**: {len(df)}")
            
            # Show LLM relationship distribution
            if llm_relationship_counts:
                relationship_df = pd.DataFrame.from_dict(llm_relationship_counts, orient='index', columns=['Count']).reset_index().rename(columns={'index': 'Relationship'})
                
                # Enhanced color mapping for different LLM relationships
                color_map = {
                    'Exact Match': '#4CAF50',
                    'Equivalent': '#4CAF50', 
                    'Related': '#FFA500',
                    'Similar': '#FFA500',
                    'Different': '#ff6b6b',
                    'Contradictory': '#ff6b6b',
                    'Below Threshold': '#9E9E9E',
                    'Error': '#9E9E9E',
                    'N/A': '#9E9E9E'
                }
                
                # Assign colors based on relationship names
                colors = []
                for rel in relationship_df['Relationship']:
                    color = color_map.get(rel, '#00d4ff')
                    # Fallback color logic for partial matches
                    if color == '#00d4ff':
                        if any(word in rel.lower() for word in ['exact', 'match', 'equivalent']):
                            color = '#4CAF50'
                        elif any(word in rel.lower() for word in ['related', 'similar', 'partial']):
                            color = '#FFA500'
                        elif any(word in rel.lower() for word in ['different', 'contradictory']):
                            color = '#ff6b6b'
                        elif any(word in rel.lower() for word in ['threshold', 'error', 'n/a']):
                            color = '#9E9E9E'
                    colors.append(color)
                
                fig = px.bar(
                    relationship_df, 
                    x='Relationship', 
                    y='Count', 
                    title="Distribution of Relationship Analysis", # Updated title
                    color='Relationship',
                    color_discrete_sequence=colors,
                    text='Count'
                )
                fig.update_traces(textposition='inside', textfont=dict(color='white', size=15, family='Arial Black'))
                fig.update_layout(
                    paper_bgcolor="#1e1e2f", 
                    plot_bgcolor="#2a2a3c", 
                    font=dict(color="white"),
                    xaxis_title="Relationship", 
                    yaxis_title="Count", 
                    showlegend=False,
                    xaxis={'tickangle': 45}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Additional LLM statistics
            if 'Remark' in df.columns:
                non_empty_remarks = df[df['Remark'].notna() & (df['Remark'] != '-') & (df['Remark'] != '')]['Remark']
                st.write(f"- **Detailed Analyses with Remarks**: {len(non_empty_remarks)}")
        else:
            # Fallback to similarity-based summary if LLM columns don't exist
            avg_similarity = df['Similarity_Score'].mean() if 'Similarity_Score' in df.columns else 0
            level_counts = df['Similarity_Level'].value_counts().to_dict() if 'Similarity_Level' in df.columns else {}
            
            st.markdown('<div class="summary-card">', unsafe_allow_html=True)
            st.markdown("**Similarity Summary** (LLM analysis not available)")
            st.write(f"- **Queries Processed**: {num_queries}")
            st.write(f"- **Average Similarity Score**: {avg_similarity:.4f}")
            
            if level_counts:
                level_df = pd.DataFrame.from_dict(level_counts, orient='index', columns=['Count']).reset_index().rename(columns={'index': 'Category'})
                fig = px.bar(
                    level_df, x='Category', y='Count', title="Distribution of Similarity Levels",
                    color='Category', color_discrete_sequence=['#00d4ff', '#ffaa00', '#00ffaa', '#ff6b6b'] * 2, text='Count'
                )
                fig.update_traces(textposition='inside', textfont=dict(color='Brown', size=15, family='Arial Black'))
                fig.update_layout(paper_bgcolor="#1e1e2f", plot_bgcolor="#2a2a3c", font=dict(color="white"),
                                  xaxis_title="Category", yaxis_title="Count", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        logging.error(f"Error displaying summary: {str(e)}")
        st.error(f"Error displaying summary: {str(e)}")