# Research Digger: Streamlining Academic Research with AI

## Table of Contents
1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [How It Works](#how-it-works)
4. [Technical Details](#technical-details)
    1. [Chain of Thought Prompting](#chain-of-thought-prompting)
5. [Usage](#usage)
6. [Future Work](#future-work)
7. [Conclusion](#conclusion)

Research Digger: https://github.com/KickItLikeShika/research-digger

## Introduction
Welcome to the official blog post for Research Digger, an intelligent tool designed to streamline the process of academic research. In this post, we will explore the motivation behind creating Research Digger, the technical details of how we built it, and the innovative concept of chain of thought prompting that powers it. We will also provide a comprehensive guide on how to use the tool and discuss future enhancements.

---

## Motivation
The motivation behind Research Digger stems from the challenges faced by researchers, students, and professionals in navigating the vast amount of academic literature available today. With the exponential growth of research papers, it has become increasingly difficult to keep up with the latest findings and extract meaningful insights efficiently. The idea for Research Digger came to me when I needed to get up to speed quickly in a specific research area and found it overwhelming to sift through numerous papers. This project aims to address these challenges by leveraging advanced AI techniques to automatically fetch, analyze, and summarize research papers. Whether you are starting research in a new area or need to stay updated with the latest developments, Research Digger makes it easier for users to stay informed and make data-driven decisions.

---

## How It Works
Research Digger is built using a combination of state-of-the-art technologies and methodologies. Here is a step-by-step overview of the development process:

1. **Literature Review**: We integrated with various academic databases and APIs, such as Semantic Scholar, to fetch relevant research papers based on user-defined topics or areas.
2. **Summarizations of Individual Papers Using LLMs**: We utilized SOTA LLMs, to analyze the content of the fetched papers and generate concise summaries.
3. **Generating the Generic Summary**: After summarizing individual papers, we use LLMs to create a comprehensive generic summary. This summary encapsulates the key insights and trends across all the fetched papers, providing users with a holistic view of the research area.

---

## Technical Details

### Chain of Thought Prompting
One of the key innovations in Research Digger is the use of chain of thought prompting. This technique involves guiding the AI model through a series of logical steps to generate more coherent and contextually relevant summaries. By breaking down the summarization process into smaller, manageable tasks, we can ensure that the generated summaries are not only accurate but also easy to understand.

For example, when summarizing a research paper, the model first identifies the main points, then elaborates on the key findings, and finally discusses the significance of those findings. This structured approach helps in producing high-quality summaries that capture the essence of the research.

---

## Usage
To use Research Digger, follow these steps:

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/KickItLikeShika/research-digger
    cd research-digger
    ```

2. **Create a Virtual Environment**:
    ```sh
    python -m venv venv
    ```

3. **Activate the Virtual Environment**:
    ```sh
    source venv/bin/activate
    ```

4. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

5. **Set Up API Keys**:
    - OpenAI API Key:
    ```sh
    export OPENAI_API_KEY='your_openai_api_key'
    ```
    - (Optional) Semantic Scholar API Key:
    ```sh
    export S2_API_KEY='your_semantic_scholar_api_key'
    ```

6. **Run the Script**:
    ```sh
    python launch_research_digger.py --research_area="Your Research Area" --papers_limit=20 --summary_length="short" --model="gpt-4o-mini"
    ```

---

## Future Work
While Research Digger is already a usable and helpful tool, there are several areas for future improvement:

1. **Model Support**: Expanding support to include more models, either commercial or open source models.
2. **Error Handling**: Enhancing error handling to provide more user-friendly messages and troubleshooting tips.
3. **Customization Options**: Adding more advanced customization options for the summarization process.
4. **User Interface**: Developing a graphical user interface (GUI) or a web-based interface to make the tool more accessible to non-technical users.

---

## Conclusion
Research Digger is a testament to the power of AI in transforming the way we conduct academic research. By automating the process of fetching, analyzing, and summarizing research papers, it empowers users to stay informed and make better decisions. We are excited about the future of Research Digger and look forward to continuing to improve and expand its capabilities.

Thank you for reading, and I hope you find Research Digger useful! Contributions are welcomed from the community to help improve Research Digger. Whether it's fixing bugs, adding new features, improving documentation, or providing feedback, your contributions are highly valued. If you have any feedback or suggestions, feel free to open an issue on GitHub. I appreciate your input and look forward to your contributions!

