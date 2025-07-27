**Intelligent Query Understanding and Augmentation**

Objective : Interpreting free-form user queries, classifying their intent, extracting relevant information, and identifying what's missing to interact with specialized "experts" (which could be different LLM prompts, APIs, or knowledge bases).

Steps to achieve the objective :
1. Classify user queries into one of these categories - CATEGORY_SEARCH, NEED_PROBLEM_BASED, INFO_GATHERING_GENERAL, INVALID_NONSENSE (query_classification.py)
2. If a query is classified as CATEGORY_SEARCH or NEED_PROBLEM_BASED, those are then classified into expert categories - Broker, Legal Advisor, Astrologer, Tech Support, Medical Consultant (expert_category.py)
3. Useful attributes are extracted using fine-tuned NER models for each category (Model training script - attribute_extraction_ner_train.py)
4. Based on certain pre-defined rules, missing attributes for the certain category are identified.
5. LLM generated follow-up questions to seek missing information from the user. (followup_generator.py)
6. A response is generated in a proper structure. 

Output structure - 

===== Query Result =====

{'original_query': 'Hi, I’m Sneha Kulkarni, a software engineer facing harassment at my Bangalore workplace. I need help understanding if I can file a complaint under Karnataka labor law.', 
 'classification': 'CATEGORY_SEARCH', 
 'identified_category': 'Legal Advisor', 
 'extracted_attributes': {'FULL_NAME': 'Sneha Kulkarni', 'OCCUPATION': 'software engineer', 'ISSUE_DESCRIPTION': 'facing harassment at my Bangalore workplace', 'JURISDICTION': 'Karnataka'}, 'missing_info': ['SUPPORTING_DOCUMENTS'], 
 'ai_questions_for_missing_info': 'Do you have any supporting documents relevant to the issue you have described?'}
 
Setup Instructions - 

1. The user should have Git LFS installed to clone the big model files too.
   In macOS - brew install git-lfs > git lfs install
   In Ubuntu - sudo apt install git-lfs > git lfs install
   In windows - Download and install Git LFS from: https://git-lfs.com > On git bash, do git lfs install

2. Clone the repository.
(bash command)
git clone <your-repo-url>
cd <your-project-folder>
Verify LFS files are downloaded -
ls -lh models/expert_classifier/
ls -lh models/query_classifier/
You should see the actual .safetensors files. If not, do git lfs pull.

3. Create a virtual environment.
(bash command)
python3 -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate # If using anaconda environment, use conda activate <virtual environment name>

4. Install dependencies.
(bash command)
pip install -r requirements.txt

5. Run main script.
(bash command)
python main.py --text "My name is Anjali Sharma. I was born on 1990-05-12 at 02:30 PM in Delhi. I identify as Female. I'm currently facing issues related to property disputes. I prefer Vedic Astrology."

Note: The first run might be slow, since the generator model will be loaded on the first run. But a local instance would be stored for all further runs.

Note: Relevant expert categories are - Broker, Legal Advisor, Astrologer, Tech Support, Medical Consultant

Note: Model MBZUAI/LaMini-Flan-T5-783M is used here for follow-up question generator to avoid memory issues. It is compatible with CPU or MPS for lightweight inference.

Note: This setup is tested on macOS using MPS backend; suitable LLMs must be chosen to avoid memory overflow.

Understanding the files structure:
- query_classification.py has the code for the classification of user queries to CATEGORY_SEARCH, NEED_PROBLEM_BASED, INFO_GATHERING_GENERAL, INVALID_NONSENSE categories.
- expert_category.py has code for classifying the user queries to one of the expert categories - Broker, Legal Advisor, Astrologer, Tech Support, Medical Consultant the the query was classified to CATEGORY_SEARCH or NEED_PROBLEM_BASED at step 1.
- attribute_extraction_ner_train.py has code to train model to extract attributes from a user query based on a certain category it has been classified into.
- ner_utils.py has some additional utility function to support attribute_extraction_ner_train.py methods.
- followup_generator.py has code to utilize an llm generator to generate followup questions for the user to seek missing information based on identified missing attributes.
- requirements.txt has all the dependencies for this project.
- main.py is the final compilation of all the fucntions, put together to generate the desired response.
- data folder has all the training data.
    - query_classification.jsonl has the data to train query_classification model.
    - expert_queries.jsonl has data to train expert_category classification model.
    - <expert_category>_ner_train_auto.jsonl has data for a particular expert category with identified attributes.
    - <expert_category>_ner_train.jsonl has the processed(annotated) data.
    - train folder has the final processed .spacy data files for all expert categories which are consumed by the ner training models.
-models folder has all the pretrained models 
    - <expert_category>_ner are ner models to extract attributes from a user query for that particular expert category.
    - query_classifier is the model for first stage query classification to ATEGORY_SEARCH, NEED_PROBLEM_BASED, INFO_GATHERING_GENERAL, INVALID_NONSENSE.
    - expert_classifier is the model to classify a user query to an expert category.

    



