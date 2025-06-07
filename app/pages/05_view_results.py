import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import datetime
from pathlib import Path
from utils.file_helpers import (
    get_results, get_result_files, get_result_file_path, read_json_file
)
from utils.session_state import load_api_keys_from_env

# Page header
st.header("📊 View Results")
st.markdown("Analyze and visualize validation results from processed papers.")

# Get available results
results = get_results()

if not results:
    st.info("📂 No results found. Please run validation first.")
    st.markdown("👉 Go to the **Run Validation** page to process papers.")
    st.stop()

# Result selection
st.subheader("📋 Select Results")

col1, col2 = st.columns([2, 1])

with col1:
    # Check if we have a recent validation result
    default_result = None
    if hasattr(st.session_state, 'last_validation_result') and st.session_state.last_validation_result:
        last_papers = st.session_state.last_validation_result.get('papers', [])
        if last_papers:
            # Try to find a result that matches the last validation
            paper_id = Path(last_papers[0]).stem.split('.')[0]
            for result in results:
                if paper_id in result:
                    default_result = result
                    break
    
    # Result selection
    if default_result and default_result in results:
        default_index = results.index(default_result)
    else:
        default_index = 0
    
    selected_result = st.selectbox(
        "Select result to view",
        results,
        index=default_index,
        help="Choose a paper result to analyze"
    )

with col2:
    # Quick stats for selected result
    if selected_result:
        result_files = get_result_files(selected_result)
        st.metric("Result Files", sum(len(files) for files in result_files.values()))

# Load and display results
if selected_result:
    st.markdown("---")
    st.subheader(f"📄 Results for: {selected_result}")
    
    # Get all files for this result
    result_files = get_result_files(selected_result)
    
    # Create tabs for different result types
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Summary", "📋 Checklist", "🧠 Reasoner", "🔍 Extractor", "✅ Validator", "📈 Analysis"
    ])
    
    with tab1:
        st.subheader("Validation Summary")
        
        # Try to load the final report
        report_data = None
        if result_files['report']:
            report_file = result_files['report'][0]  # Get the most recent
            report_path = get_result_file_path(selected_result, report_file)
            report_data = read_json_file(report_path)
        
        if report_data:
            # Display summary metrics
            if 'validation_summary' in report_data:
                metrics = report_data['validation_summary']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Items", metrics.get('total_items', 0))
                
                with col2:
                    agreement = metrics.get('agree_with_extractor', 0)
                    st.metric("Agreements", agreement)
                
                with col3:
                    disagreement = metrics.get('disagree_with_extractor', 0)
                    st.metric("Disagreements", disagreement)
                
                with col4:
                    agreement_rate = metrics.get('agreement_rate', 0)
                    st.metric("Agreement Rate", f"{agreement_rate:.1f}%")
            
            # Paper and checklist info
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Paper Information:**")
                st.write(f"Paper: {report_data.get('paper', 'Unknown')}")
                st.write(f"Checklist: {report_data.get('checklist', 'Unknown')}")
                
                # Model information if available
                if 'model_info' in report_data:
                    st.markdown("**Model Information:**")
                    model_info = report_data['model_info']
                    for agent, model in model_info.items():
                        st.write(f"{agent.title()}: {model}")
            
            with col2:
                # Compliance distribution
                if 'items' in report_data:
                    compliance_counts = {}
                    for item_data in report_data['items'].values():
                        compliance = item_data.get('compliance', 'unknown')
                        compliance_counts[compliance] = compliance_counts.get(compliance, 0) + 1
                    
                    if compliance_counts:
                        fig = px.pie(
                            values=list(compliance_counts.values()),
                            names=list(compliance_counts.keys()),
                            title="Compliance Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No summary report found for this result.")
    
    with tab2:
        st.subheader("Final Checklist")
        
        # Try to load the checklist file
        checklist_data = None
        if result_files['checklist']:
            checklist_file = result_files['checklist'][0]  # Get the most recent
            checklist_path = get_result_file_path(selected_result, checklist_file)
            checklist_data = read_json_file(checklist_path)
        
        if checklist_data and 'checklist' in checklist_data:
            # Create a DataFrame for better display
            checklist_items = []
            for item_id, item_data in checklist_data['checklist'].items():
                checklist_items.append({
                    'Item ID': item_id,
                    'Description': item_data.get('description', '')[:100] + '...' if len(item_data.get('description', '')) > 100 else item_data.get('description', ''),
                    'Answer': item_data.get('answer', 'No answer provided')
                })
            
            if checklist_items:
                df = pd.DataFrame(checklist_items)
                
                # Add search functionality
                search_term = st.text_input("🔍 Search checklist items", placeholder="Enter item ID or description")
                
                if search_term:
                    mask = df.apply(lambda x: x.astype(str).str.contains(search_term, case=False, na=False)).any(axis=1)
                    filtered_df = df[mask]
                else:
                    filtered_df = df
                
                # Display the table
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download option
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Checklist as CSV",
                    data=csv,
                    file_name=f"{selected_result}_checklist.csv",
                    mime="text/csv"
                )
            else:
                st.info("📝 No checklist items found.")
        else:
            st.warning("⚠️ No checklist file found for this result.")
    
    with tab3:
        st.subheader("Reasoner Output")
        
        if result_files['reasoner']:
            reasoner_file = result_files['reasoner'][0]
            reasoner_path = get_result_file_path(selected_result, reasoner_file)
            reasoner_data = read_json_file(reasoner_path)
            
            if reasoner_data:
                st.markdown(f"**File:** {reasoner_file}")
                
                # Show prompts count
                if isinstance(reasoner_data, dict):
                    st.info(f"📊 Generated {len(reasoner_data)} prompts")
                    
                    # Search functionality
                    search_prompt = st.text_input("🔍 Search prompts", placeholder="Enter item ID or prompt text")
                    
                    # Display prompts
                    for item_id, prompt in reasoner_data.items():
                        if not search_prompt or search_prompt.lower() in item_id.lower() or search_prompt.lower() in str(prompt).lower():
                            with st.expander(f"📝 Item {item_id}"):
                                if isinstance(prompt, str):
                                    st.text_area("Prompt", prompt, height=150, key=f"reasoner_prompt_{item_id}")
                                else:
                                    st.json(prompt)
                else:
                    st.json(reasoner_data)
            else:
                st.error("❌ Failed to load reasoner data")
        else:
            st.info("📝 No reasoner output found for this result.")
    
    with tab4:
        st.subheader("Extractor Output")
        
        if result_files['extractor']:
            extractor_file = result_files['extractor'][0]
            extractor_path = get_result_file_path(selected_result, extractor_file)
            extractor_data = read_json_file(extractor_path)
            
            if extractor_data:
                st.markdown(f"**File:** {extractor_file}")
                
                # Show extraction count
                if isinstance(extractor_data, dict):
                    st.info(f"📊 Processed {len(extractor_data)} items")
                    
                    # Search functionality
                    search_extraction = st.text_input("🔍 Search extractions", placeholder="Enter item ID or content")
                    
                    # Display extractions
                    for item_id, extraction in extractor_data.items():
                        if not search_extraction or search_extraction.lower() in item_id.lower() or search_extraction.lower() in str(extraction).lower():
                            with st.expander(f"🔍 Item {item_id}"):
                                if isinstance(extraction, dict) and 'extracted_content' in extraction:
                                    content = extraction['extracted_content']
                                    
                                    # Show item description first if available
                                    item_description = content.get('item_description', extraction.get('item_description', ''))
                                    if item_description:
                                        st.markdown(f"**Item Description:** {item_description}")
                                        st.markdown("---")
                                    
                                    # Display compliance and answer
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        compliance = content.get('compliance', 'Unknown')
                                        st.write(f"**Compliance:** {compliance}")
                                        
                                        confidence = content.get('confidence', '')
                                        if confidence:
                                            st.write(f"**Confidence:** {confidence}")
                                        
                                        reasoning = content.get('reasoning', 'No reasoning provided')
                                        st.write(f"**Reasoning:** {reasoning}")
                                    
                                    with col2:
                                        # Show the extracted answer
                                        answer = content.get('answer', content.get('correct_answer', 'No answer provided'))
                                        st.write(f"**Answer:** {answer}")
                                        
                                        # Show item ID if available
                                        item_id_field = content.get('item_id', extraction.get('item_id', item_id))
                                        st.write(f"**Item ID:** {item_id_field}")
                                    
                                    # Evidence section (full width)
                                    st.markdown("**Evidence:**")
                                    evidence = content.get('evidence', [])
                                    if evidence:
                                        for i, ev in enumerate(evidence):
                                            if isinstance(ev, dict):
                                                quote = ev.get('quote', 'No quote')
                                                location = ev.get('location', '')
                                                if location:
                                                    st.write(f"{i+1}. \"{quote}\" *(Location: {location})*")
                                                else:
                                                    st.write(f"{i+1}. \"{quote}\"")
                                            else:
                                                st.write(f"{i+1}. {ev}")
                                    else:
                                        st.write("*No evidence found*")
                                else:
                                    st.json(extraction)
                else:
                    st.json(extractor_data)
            else:
                st.error("❌ Failed to load extractor data")
        else:
            st.info("📝 No extractor output found for this result.")
    
    with tab5:
        st.subheader("Validator Output")
        
        if result_files['validator']:
            validator_file = result_files['validator'][0]
            validator_path = get_result_file_path(selected_result, validator_file)
            validator_data = read_json_file(validator_path)
            
            if validator_data:
                st.markdown(f"**File:** {validator_file}")
                
                # Show validation count
                if isinstance(validator_data, dict):
                    st.info(f"📊 Validated {len(validator_data)} items")
                    
                    # Validation summary
                    validation_counts = {}
                    for validation in validator_data.values():
                        if isinstance(validation, dict):
                            result = validation.get('validate_result', 'unknown')
                            validation_counts[result] = validation_counts.get(result, 0) + 1
                    
                    if validation_counts:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Agreements", validation_counts.get('agree with extractor', 0))
                        
                        with col2:
                            st.metric("Disagreements", validation_counts.get('do not agree with extractor', 0))
                        
                        with col3:
                            st.metric("Unknown", validation_counts.get('unknown', 0))
                    
                    # Search functionality
                    search_validation = st.text_input("🔍 Search validations", placeholder="Enter item ID or content")
                    
                    # Display validations
                    for item_id, validation in validator_data.items():
                        if not search_validation or search_validation.lower() in item_id.lower() or search_validation.lower() in str(validation).lower():
                            with st.expander(f"✅ Item {item_id}"):
                                if isinstance(validation, dict):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write(f"**Validation Result:** {validation.get('validate_result', 'Unknown')}")
                                        st.write(f"**Correct Answer:** {validation.get('correct_answer', 'No answer provided')}")
                                    
                                    with col2:
                                        st.write(f"**Reason:** {validation.get('Reason', 'No reason provided')}")
                                else:
                                    st.json(validation)
                else:
                    st.json(validator_data)
            else:
                st.error("❌ Failed to load validator data")
        else:
            st.info("📝 No validator output found for this result.")
    
    with tab6:
        st.subheader("Detailed Analysis")
        
        # Load all data for comprehensive analysis
        report_data = None
        if result_files['report']:
            report_file = result_files['report'][0]
            report_path = get_result_file_path(selected_result, report_file)
            report_data = read_json_file(report_path)
        
        if report_data and 'items' in report_data:
            items_data = report_data['items']
            
            # Create analysis DataFrame
            analysis_data = []
            for item_id, item_info in items_data.items():
                analysis_data.append({
                    'Item ID': item_id,
                    'Description': item_info.get('description', '')[:50] + '...' if len(item_info.get('description', '')) > 50 else item_info.get('description', ''),
                    'Compliance': item_info.get('compliance', 'unknown'),
                    'Evidence Count': len(item_info.get('evidence', [])),
                    'Has Reasoning': 'Yes' if item_info.get('reasoning', '') else 'No',
                    'Final Answer': item_info.get('correct_answer', 'No answer')[:50] + '...' if len(item_info.get('correct_answer', '')) > 50 else item_info.get('correct_answer', 'No answer')
                })
            
            if analysis_data:
                df_analysis = pd.DataFrame(analysis_data)
                
                # Compliance analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    # Compliance distribution chart
                    compliance_counts = df_analysis['Compliance'].value_counts()
                    fig = px.bar(
                        x=compliance_counts.index,
                        y=compliance_counts.values,
                        title="Compliance Distribution",
                        labels={'x': 'Compliance Status', 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Evidence distribution
                    evidence_counts = df_analysis['Evidence Count'].value_counts().sort_index()
                    fig = px.bar(
                        x=evidence_counts.index,
                        y=evidence_counts.values,
                        title="Evidence Count Distribution",
                        labels={'x': 'Number of Evidence Items', 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table
                st.markdown("### Detailed Item Analysis")
                
                # Filters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    compliance_filter = st.multiselect(
                        "Filter by Compliance",
                        options=df_analysis['Compliance'].unique(),
                        default=df_analysis['Compliance'].unique()
                    )
                
                with col2:
                    max_evidence = int(df_analysis['Evidence Count'].max()) if not df_analysis.empty and df_analysis['Evidence Count'].max() > 0 else 1
                    evidence_filter = st.slider(
                        "Minimum Evidence Count",
                        min_value=0,
                        max_value=max_evidence,
                        value=0
                    )
                
                with col3:
                    reasoning_filter = st.selectbox(
                        "Has Reasoning",
                        options=['All', 'Yes', 'No'],
                        index=0
                    )
                
                # Apply filters
                filtered_df = df_analysis[
                    (df_analysis['Compliance'].isin(compliance_filter)) &
                    (df_analysis['Evidence Count'] >= evidence_filter)
                ]
                
                if reasoning_filter != 'All':
                    filtered_df = filtered_df[filtered_df['Has Reasoning'] == reasoning_filter]
                
                # Display filtered table
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Export options
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_analysis = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Analysis as CSV",
                        data=csv_analysis,
                        file_name=f"{selected_result}_analysis.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    if st.button("📊 Generate Report Summary"):
                        st.markdown("### Summary Report")
                        st.write(f"**Total Items Analyzed:** {len(df_analysis)}")
                        st.write(f"**Items with Evidence:** {len(df_analysis[df_analysis['Evidence Count'] > 0])}")
                        st.write(f"**Items with Reasoning:** {len(df_analysis[df_analysis['Has Reasoning'] == 'Yes'])}")
                        
                        compliance_summary = df_analysis['Compliance'].value_counts()
                        st.write("**Compliance Summary:**")
                        for status, count in compliance_summary.items():
                            percentage = (count / len(df_analysis)) * 100
                            st.write(f"  - {status}: {count} ({percentage:.1f}%)")
            else:
                st.info("📝 No analysis data available.")
        else:
            st.warning("⚠️ No detailed analysis data found for this result.")

# Batch analysis for multiple results
if len(results) > 1:
    st.markdown("---")
    st.subheader("📈 Batch Analysis")
    
    with st.expander("Compare Multiple Results"):
        selected_results = st.multiselect(
            "Select results to compare",
            results,
            default=results[:3] if len(results) >= 3 else results
        )
        
        if len(selected_results) > 1:
            comparison_data = []
            
            for result in selected_results:
                result_files = get_result_files(result)
                if result_files['report']:
                    report_file = result_files['report'][0]
                    report_path = get_result_file_path(result, report_file)
                    report_data = read_json_file(report_path)
                    
                    if report_data and 'validation_summary' in report_data:
                        metrics = report_data['validation_summary']
                        comparison_data.append({
                            'Paper': result,
                            'Total Items': metrics.get('total_items', 0),
                            'Agreements': metrics.get('agree_with_extractor', 0),
                            'Disagreements': metrics.get('disagree_with_extractor', 0),
                            'Agreement Rate (%)': metrics.get('agreement_rate', 0),
                            'Checklist': report_data.get('checklist', 'Unknown')
                        })
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                
                # Display comparison table
                st.dataframe(df_comparison, use_container_width=True, hide_index=True)
                
                # Comparison charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        df_comparison,
                        x='Paper',
                        y='Agreement Rate (%)',
                        title="Agreement Rate Comparison",
                        color='Checklist'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.scatter(
                        df_comparison,
                        x='Total Items',
                        y='Agreement Rate (%)',
                        size='Agreements',
                        color='Checklist',
                        title="Items vs Agreement Rate",
                        hover_data=['Paper']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No comparable data found for selected results.")

# Quick action for FHIR Evidence Generation
if selected_result and 'report_data' in locals() and report_data:
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("**🔬 FHIR Evidence Generation Available**")
        st.markdown("Generate standardized FHIR Evidence resources for clinical research interoperability.")
    
    with col2:
        if st.button("🔬 Go to FHIR Generator", type="primary"):
            st.info("👉 Navigate to the **FHIR Evidence** page to generate FHIR resources.")

# Help section
with st.expander("💡 Understanding Results"):
    st.markdown("""
    **Result Components:**
    - **Summary**: Overview of validation metrics and compliance distribution
    - **Checklist**: Final answers for each checklist item
    - **Reasoner**: Generated prompts from guideline analysis
    - **Extractor**: Information extracted from the paper
    - **Validator**: Validation of extracted information
    - **Analysis**: Detailed statistical analysis and visualizations
    
    **Key Metrics:**
    - **Agreement Rate**: Percentage of items where validator agrees with extractor
    - **Evidence Count**: Number of supporting quotes found for each item
    - **Compliance Status**: Whether the paper meets each checklist requirement
    
    **Interpreting Results:**
    - High agreement rates indicate consistent analysis
    - Multiple evidence items suggest thorough analysis
    - "Unknown" compliance may indicate unclear reporting in the paper
    - Disagreements highlight items needing manual review
    
    **Next Steps:**
    - Use the **FHIR Evidence** page to generate standardized evidence resources
    - Export results as CSV for further analysis
    - Compare multiple papers using the batch analysis feature
    """)
