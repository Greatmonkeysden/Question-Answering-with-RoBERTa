import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

def main():
    st.title("Question Answering with RoBERTa")

    # Load model & tokenizer
    model_name = "deepset/roberta-base-squad2"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get user input
    context = st.text_area("Enter the context (max 400 words):")
    question = st.text_input("Enter your question:")


    if st.button("Get Answer"):
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
                    if i==1 or i==0:
                      continue
                    else:
                      st.write(f" Answer: {answer}") #Top-{i + 1}

            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
