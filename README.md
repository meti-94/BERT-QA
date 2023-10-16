# BERT-QA: Question Answering with BERT over Knowledge Graphs

BERT-QA is an advanced academic project that employs language models for returning relevant answers to natural language questions. The process involves:

1. Extracting the queried entity and linking it to a unique machine ID.
2. Recognizing the relation being questioned about the entity.
3. Using the aforementioned information to query on a knowledge graph.
4. Retrieving the answer based on the graph's data.

## Publication

The repository is related to the research article:
- **"Improved relation span detection in question answering systems over extracted knowledge bases"**  
  [Read the full article here](https://www.sciencedirect.com/science/article/pii/S095741742300475X).

## Installation and Requirements

### Prerequisites

- Operating System: LINUX or WINDOWS
- Python Version: 3.7

### Dependencies

The following packages are required:

```
transformers==4.6.0
torch==1.8.1+cu101
networkx==2.5
```

### Installation

After setting up your environment with the prerequisites, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the repository directory.
3. Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Workflow Diagram


To understand the training and inference workflow, please refer to the diagram provided below:

![BERT-QA Workflow Diagram](rsc/OV.jpg).

## Contribution & Contact

- **Code-related Queries**: For any questions related to the code or if you're interested in contributing, please reach out to Mehdi at [mehdi_j94@outlook.com](mailto:mehdi_j94@outlook.com).

- **Dataset Access**: To get access to the dataset used in this project, please contact Somayyeh Behmanesh at [somayyeh.behmanesh@gmail.com](mailto:somayyeh.behmanesh@gmail.com).

## License

This project is open source. Please make sure to credit the authors if you use any part of this work in your research or project.

