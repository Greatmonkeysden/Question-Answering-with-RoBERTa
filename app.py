import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

def main():
    # Custom CSS for styling
    custom_css = """
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #FF0000; /* Set your desired background color here */
            }
            .title {
                font-size: 2.5em;
                color: #333;
                text-align: center;
                padding: 1em;
                background-color: #3498db;
                color: #fff;
                border-radius: 10px;
                margin-bottom: 1em;
            }
            .input-container {
                margin: 2em;
            }
            .button-container {
                text-align: center;
            }
            .result-container {
                margin: 2em;
                padding: 1em;
                background-color: #fff;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .answer {
                background-color: #3498db;
                padding: 10px;
                border-radius: 5px;
                margin-top: 10px;
            }
            img {
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    # Title
    st.markdown("<div class='title'>Question Answering with Transformers</div>", unsafe_allow_html=True)

    # Model Selection Dropdown
    model_name = st.selectbox("Select Model", ["deepset/roberta-base-squad2", "bert-large-uncased-whole-word-masking-finetuned-squad", "distilbert-base-cased-distilled-squad",
    "bert-base-uncased",
    "albert-base-v2"])
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get user input
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    context = st.text_area("Enter the context (max 400 words):")
    question = st.text_input("Enter your question:")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Get Answer"):
        #st.markdown("<div class='result-container'>", unsafe_allow_html=True)
        if not question or not context:
            st.warning("Please enter both a question and a context.")
        else:
            # Tokenize input
            try:
                # Check word count in the context
                if len(context.split()) > 400:
                    raise ValueError("Context exceeds 400 words limit.")

                inputs = tokenizer.encode_plus(question, context, return_tensors='pt')

                # Get predictions
                outputs = model(**inputs)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits

                # Get top N answer spans
                top_n = 3
                start_indexes = start_logits.argsort(dim=1, descending=True)[:, :top_n]
                end_indexes = end_logits.argsort(dim=1, descending=True)[:, :top_n]

                # Display detailed answers
                st.subheader(f"Question: {question}")
                for i in range(top_n):
                    start_index = start_indexes[0, i].item()
                    end_index = end_indexes[0, i].item()
                    answer = tokenizer.decode(inputs['input_ids'][0, start_index:end_index + 1])

                    # Highlight answer in context
                    highlighted_context = f"{context[:start_index]}**{context[start_index:end_index+1]}**{context[end_index+1:]}"

                    # Display confidence scores
                    confidence_start = start_logits[0, start_index].item()
                    confidence_end = end_logits[0, end_index].item()
                    if answer == "":
                        continue
                    else:
                      st.markdown(f"<div class='answer'><strong>Answer:</strong> {answer}<br>"
                                f"<strong>Confidence (Start):</strong> {confidence_start:.4f}<br>"
                                f"<strong>Confidence (End):</strong> {confidence_end:.4f}</div>", unsafe_allow_html=True)

            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.error(f"An error occurred: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
