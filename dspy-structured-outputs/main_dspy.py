import os
import argparse
import dspy
import logging
from groq import Groq
from dotenv import load_dotenv
from executive_summary_pipeline import ExecutiveSummaryPipeline

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_environment():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return api_key

def configure_groq_client(api_key):
    client = Groq(api_key=api_key)
    model_name = "llama3-8b-8192"
    return client, model_name

def save_to_markdown(filename, content):
    with open(filename, 'w') as file:
        file.write(content)

def main():
    parser = argparse.ArgumentParser(description="Process LLM responses and format them.")
    parser.add_argument(
        '-m', '--markdown', action='store_true',
        help="Save the output in markdown format."
    )
    args = parser.parse_args()

    try:
        api_key = load_environment()
        groq_client, model_name = configure_groq_client(api_key)

        # Configure the Groq client as the default LM for DSPy without cache
        dspy.configure(
            lm=dspy.OpenAI(
                client=groq_client,
                model=model_name,
                temperature=0.7,
                max_tokens=2048,  # Ensure max_tokens is sufficiently high
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            ),
            cache=False  # Disable caching explicitly if supported by the framework
        )

        # Simplified prompt for Executive Summary
        prompt_exec_summary = "Explain the benefits and trade-offs of using machine learning in healthcare."
        
        logging.debug(f"Prompt for Executive Summary: {prompt_exec_summary}")
        
        exec_summary_pipeline = ExecutiveSummaryPipeline()
        
        try:
            formatted_exec_summary = exec_summary_pipeline(prompt_exec_summary)
            logging.debug(f"Formatted Executive Summary: {formatted_exec_summary}")
        except Exception as e:
            logging.error(f"Error in Executive Summary Pipeline: {e}")
            raise

        # Output results
        print("Executive Summary:")
        print(formatted_exec_summary)

        # Save to markdown if the flag is set
        if args.markdown:
            save_to_markdown('executive_summary.md', formatted_exec_summary)
            logging.info("Markdown file saved as 'executive_summary.md'")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()