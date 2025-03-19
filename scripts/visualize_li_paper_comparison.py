import pandas as pd
import plotly.graph_objects as go

# Load the comparison data
comparison_df = pd.read_csv('output/li_paper_comparison.csv')

# Prepare data for Sankey diagram
source = [0, 1]
target = [1, 0]
value = [0, 0]

# Iterate over each row to populate the Sankey diagram data
for _, row in comparison_df.iterrows():
    if row.iloc[2] == row.iloc[3]:  # Assuming the first 'correct_answer' is for Claude and the second is for OpenAI
        source.append('Agreement')
        target.append('Agreement')
        value[0] += 1
    else:
        source.append('Disagreement')
        target.append('Disagreement')
        value[1] += 1

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=["Agreement", "Disagreement"]
    ),
    link=dict(
        source=source,
        target=target,
        value=value
    )
)])

# Update layout
fig.update_layout(title_text="Agreement Flows Between Extractors and Validators", font_size=10)

# Save the figure
fig.write_html('output/sankey_diagram.html')
