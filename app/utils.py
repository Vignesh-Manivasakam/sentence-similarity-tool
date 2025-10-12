import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
from app.preprocess import preprocess_data
import logging

def excel_to_json(file_obj, tokenizer, id_col, text_col, meta_cols=None):
    try:
        if meta_cols is None:
            meta_cols = []

        file_obj.seek(0)
        df = pd.read_excel(file_obj, dtype={text_col: str, id_col: str})
        
        all_cols = [id_col, text_col] + meta_cols
        if not all(c in df.columns for c in all_cols):
            missing_cols = [c for c in all_cols if c not in df.columns]
            raise ValueError(f"Selected columns {missing_cols} not found in Excel file")
        
        df[text_col] = df[text_col].fillna('').astype(str)
        df[id_col] = df[id_col].fillna('').astype(str)
        
        # Additional validation for text column
        if df[text_col].str.strip().eq('').all():
            raise ValueError("Text column contains only empty or whitespace values")
        
        logging.info(f"Excel file processed: {len(df)} rows")
        logging.debug(f"Excel columns: {df.columns.tolist()}")
        logging.debug(f"Sample data: {df.head(2).to_dict()}")
        
        # Select and rename columns
        df_subset = df[[id_col, text_col] + meta_cols].copy()
        rename_map = {id_col: 'Object_Identifier', text_col: 'Object_Text'}
        df_subset.rename(columns=rename_map, inplace=True)
        
        # Convert to dictionary records
        data = df_subset.to_dict(orient='records')
        
        processed_data, skipped_empty_count = preprocess_data(data, tokenizer)
        
        logging.info(f"Preprocessed {len(processed_data)} entries, skipped {skipped_empty_count} empty texts")
        return processed_data, skipped_empty_count
        
    except Exception as e:
        logging.error(f"Failed to process Excel file: {str(e)}")
        raise ValueError(f"Error processing Excel file: {str(e)}")

def plot_embeddings(base_embeddings, user_embeddings, base_data, user_data):
    """Generate 3D embedding visualization with smart duplicate handling."""
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
            categories = ['Base'] * len(base_embeddings)
            texts = [entry['Original_Text'] for entry in base_data]
            
            # Add note about exact matches
            title = "3D Embedding Visualization - Base Only (All queries are exact matches)"
        else:
            # Combine base and filtered user embeddings
            combined_embeddings = np.vstack((base_embeddings, filtered_user_embeddings))
            base_count = len(base_embeddings)
            user_count = len(filtered_user_embeddings)
            categories = ['Base'] * base_count + ['Query'] * user_count
            texts = ([entry['Original_Text'] for entry in base_data] + 
                    [entry['Original_Text'] for entry in filtered_user_data])
            
            title = f"3D Embedding Visualization ({len(filtered_user_embeddings)} unique queries shown)"
        
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(combined_embeddings)
        
        # Color mapping based on whether we're showing both types
        if show_only_base:
            color_map = {'Base': '#00d4ff'}
        else:
            color_map = {'Base': '#00d4ff', 'Query': '#ffaa00'}
        
        fig = px.scatter_3d(
            x=reduced[:, 0], 
            y=reduced[:, 1], 
            z=reduced[:, 2],
            color=categories, 
            color_discrete_map=color_map,
            hover_data={'text': texts, 'category': categories},
            title=title,
            labels={'x': 'Main Pattern', 'y': 'Secondary Pattern', 'z': 'Tertiary Pattern'},
            opacity=0.8,
            size=[10] * len(reduced),
            size_max=15
        )
        
        fig.update_traces(
            hovertemplate="%{customdata[0]}<extra></extra>",
            marker=dict(line=dict(width=0))
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title="Main Pattern", 
                yaxis_title="Secondary Pattern",
                zaxis_title="Tertiary Pattern", 
                bgcolor="#1e1e2f"
            ),
            paper_bgcolor="#1e1e2f", 
            font=dict(color="white"),
            clickmode='event+select', 
            legend=dict(title="Embedding Type", font=dict(color="white")),
            width=1000,
            height=640,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig
        
    except Exception as e:
        logging.error(f"Error creating embedding plot: {str(e)}")
        fig = px.scatter_3d(title="Embedding Visualization (Error occurred)")
        return fig