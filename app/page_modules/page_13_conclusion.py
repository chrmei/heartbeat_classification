"""
Page 12: Conclusion
Todo by Kiki
"""

import streamlit as st


def render():
    st.title("Conclusion")
    st.markdown("---")

    st.header("Project Summary")

    # TODO by Kiki: Add project summary
    st.subheader("Content Placeholder")
    st.write(
        """
    **TODO by Kiki:**
    - Both classification tasks were solved successfully
    - Better performance than in [2]
    - Number of misclassifications is small, but needs to be improved
    """
    )

    st.header("Results Linking to Business Issue")

    # TODO by Kiki: Add business issue linking
    st.subheader("Content Placeholder")
    st.write(
        """
    **TODO by Kiki:**
    - Link results to the business issue
    - Explain how automated ECG signal analysis pipeline could be developed for clinical contexts
    - Classify ECG signals as normal or abnormal
    - Provide initial identification of specific arrhythmias or MI
    - Medical experts would verify preliminary assessments
    - Value for rapid initial evaluations and accelerated clinical decision-making
    """
    )

    st.header("Critical Considerations")

    # TODO by Kiki: Add critical considerations
    st.subheader("Content Placeholder")
    st.write(
        """
    **TODO by Kiki:**
    - Model results should serve as decision-making tool, not definitive diagnosis
    - In-depth review and validation by qualified medical professionals is essential
    - Clinically valuable and ethically essential for patient safety
    - Maintain appropriate standard of medical care
    """
    )

    st.header("Criticism and Outlook")

    # TODO by Kiki: Add criticism and outlook
    st.subheader("Content Placeholder")
    st.write(
        """
    **TODO by Kiki:**
    - Increase quality of datasets regarding class balance
    - Reduce impact of chosen oversampling method on model performance
    - MIT dataset: especially for classes 1 and 3, additional data should be added
    - Class 0: present labels should be reviewed by medical experts, particularly for samples with extreme RR distance values
    - Test whether changes increase model performance and minimize misclassifications
    - Important for misclassifications belonging to true labels 1-4, predicted as class 0 (false negatives)
    - False negatives are particularly dangerous in clinical contexts
    - For DL model retrained on PTB: number of misclassifications is low
    - Pre-trained model may compensate for class imbalances
    - Priority: increase dataset quality for MIT dataset
    """
    )

    st.header("Future Work")

    # TODO by Kiki: Add future work
    st.subheader("Content Placeholder")
    st.write(
        """
    **TODO by Kiki:**
    - What could have been done with more time
    - Potential improvements and extensions
    - Further research directions
    """
    )

    st.header("Citations")

    # TODO by Kiki: Add citations
    st.subheader("Content Placeholder")
    st.write(
        """
    **TODO by Kiki:**
    - Add citations from Presentation.txt
    - Format references properly
    """
    )
