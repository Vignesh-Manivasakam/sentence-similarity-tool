# Image Prompt: AI Requirement Similarity Assistant (AIS Assist)

## Objective
Illustrate the dual-path processing logic of the Requirement Similarity Assistant, highlighting the "Exact Match" bypass shortcut (token saver) and the feedback database loop.

## Image Style
Modern cybernetic data-flow diagram, tech dashboard style. High-contrast neon green and purple light-paths on a dark obsidian background.

## Composition
A horizontal data-flow schematic. Text requirements enter from the left.
- **Top Path**: Flows directly to a green "Exact String Match" bypass module (labeled "0 LLM Cost").
- **Bottom Path**: Flows into a ChromaDB database block, checks against previous reviews (Short-Term Memory), and routes to an NVIDIA NIM LLM Node (`meta/llama-3.1-70b-instruct`) for semantic classification.
- All paths converge on an interactive review UI on the right showing highlighted differences.

## Color Palette
- Deep Slate/Obsidian: `#0B0F19`
- Neon Cyan: `#06B6D4`
- Neon Green: `#10B981`
- Vibrant Purple: `#8B5CF6`

## Dimensions
1200 x 600 pixels

## Detailed Prompt for ChatGPT/DALL-E 3
`An abstract data routing circuit on a dark micro-grid background. Text data enters and immediately hits a decision gate: one path branches off directly to a green 'Exact Match' shortcut (bypassing the model), while the other path flows into a vector database stack, checks against previous review feedback cache, and triggers a purple LLM query node. Sleek data-flow aesthetic, glowing neon green and purple light-paths, high-contrast engineering illustration.`
