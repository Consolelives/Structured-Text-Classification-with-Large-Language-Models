from openai import OpenAI
import pandas as pd
import json
import os
import requests
from dotenv import load_dotenv
import google.generativeai
import anthropic
from pydantic import BaseModel, Field
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import List, Optional, Union
import instructor
from datetime import date, datetime
from dateutil import parser
import re
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import ast


load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')



"""
This code defines a structured data model for classifying texts using Pydantic and Python's Enum class. 
It specifies categories of business, sports, and entertainment into subcategories.
This structure ensures data consistency, enables automatic validation, and facilitates easy integration with AI models.
"""

class BusinessCategory(str, Enum):
    STOCK_MARKET = "stock_market"
    COMPANY_NEWS = "company_news"
    ECONOMY = "economy"
    PERSONAL_FINANCE = "personal_finance"
    PROPERTY = "property"
    TECHNOLOGY_MEDIA = "technology_media"
    RETAIL_ONLINE_RETAIL = "retail_online_retail"
    SMALL_BUSINESS = "small_business"
    ANALYSIS_FEATURES = "analysis_features"
    AUDIO_PROGRAMMES = "audio_programmes"
    MERGERS_AND_ACQUISITIONS = "mergers_and_acquisitions"
    ENERGY_OIL_GAS = "energy_oil_gas"
    OTHER = "other"

class EntertainmentCategory(str, Enum):
    CINEMA = "cinema"
    THEATRE = "theatre"
    MUSIC = "music"
    LITERATURE = "literature"
    TELEVISION = "television"
    OTHER = "other"

class SportsCategory(str, Enum):
    FOOTBALL = "football"
    CRICKET = "cricket"
    RUGBY_UNION = "rugby_union"
    RUGBY_LEAGUE = "rugby_league"
    TENNIS = "tennis"
    GOLF = "golf"
    FORMULA_1 = "formula_1"
    ATHLETICS = "athletics"
    CYCLING = "cycling"
    OLYMPICS = "olympics"
    BOXING = "boxing"
    SNOW_SPORTS = "snow_sports"
    OTHER_SPORTS = "other_sports"


class NamedEntity(BaseModel):
    name: str = Field(description="Full name of the identified media personality")
    job: str = Field(description="Type of media job, e.g., Politician, Musician, Footballer etc.")

class EventSummary(BaseModel):
    event_date: str = Field(description="Event dates in April as a string, format strictly enforced")
    title: Optional[str] = None
    description: Optional[str] = None

class CategoryClassification(BaseModel):
    business: Optional[BusinessCategory] = None
    sports: Optional[SportsCategory] = None
    entertainment: Optional[EntertainmentCategory] = None
    confidence: float = Field(ge=0, le=1, description="Confidence score for the classification")
    named_entities: List[NamedEntity] = Field(default_factory=list, description="Extracted media personalities and their job types")
    april_events: List[EventSummary] = Field(default_factory=list, description="List of events that took place only in April or are scheduled for April. Do not return any other month")



class BBC:

    openai = OpenAI()
    client = instructor.patch(openai)
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    def __init__(self, root_dir, system_content, model, trained_df=None, train=False):
        self.root_dir = root_dir
        self.system_content = system_content
        self.model = model
        self.train = train
        self.trained_df = trained_df


    def clean_data(self):
        if self.train == True:
            print('Start Cleaning Data')
            # Prepare a list to collect data
            data = []
            
            # Iterate through each class/folder
            for class_name in os.listdir(self.root_dir):
                class_path = os.path.join(self.root_dir, class_name)
                if os.path.isdir(class_path):
                    # Iterate through each text file in the folder
                    for filename in os.listdir(class_path):
                        if filename.endswith(".txt"):
                            file_path = os.path.join(class_path, filename)
                            with open(file_path, 'r', encoding='utf-8') as file:
                                text = file.read().strip()
                                nos = os.path.splitext(filename)[0]
                                data.append({'nos': nos, 'text': text, 'classes': class_name})
    
            # Create DataFrame
            self.df = pd.DataFrame(data, columns=['nos', 'text', 'classes'])
            self.df = self.df.query("classes in ['business', 'sport', 'entertainment']")
    
            # Get words, character counts, remove spaces etc
            self.df['text'] = self.df.text.apply(lambda x: re.sub(r'\n\n', ' ', x))
            self.df['char_counts'] = self.df.text.apply(lambda x: len(re.sub(r'\s', '', x)))
            self.df['word_counts'] = self.df.text.apply(lambda x: len(x.split()))
            self.df['avg_word_len'] = round(self.df.char_counts/self.df.word_counts, 1)
            self.df = self.df.query("classes in ['business', 'sport', 'entertainment']")
    
            # Example: print or save
            display(self.df.head())
            self.df.to_csv(f'bbc_{self.current_date}.csv', index=False)
            self.df = pd.read_csv(f'bbc_{self.current_date}.csv')
            print('End Cleaning Data')
            print()
            print()
            print('=================================================================================================================================')
        else:
            print('Continue with Trained_df')

            
    def classify_category(self, text: str) -> CategoryClassification:
        response = self.client.chat.completions.create(
            model= self.model,
            response_model= CategoryClassification,
            temperature=0,
            max_retries=3,
            messages=[
                {
                    "role": "system",
                    "content": self.system_content
                },
                {"role": "user", "content": text}
            ]
        )
        return response

    def run_model(self):
        
        if self.train == True:
            print(f'Start Training with {self.model}')
            start = time.time()
            
            results = []
            for value in self.df['text']:
                result1 = self.classify_category(text = value)
                result1 = result1.model_dump_json()
                results.append(result1)
                
            # Step 1: Parse JSON strings into dictionaries (if they are strings)
            parsed_data = [json.loads(row) if isinstance(row, str) else row for row in results]
            
            # Step 2: Create DataFrame directly from list of dicts
            df_mini = pd.DataFrame(parsed_data)
            
            end = time.time()
            print(f"Total time: {end - start:.2f} seconds")
    
            self.df = pd.concat([self.df.reset_index(drop=True), df_mini.reset_index(drop=True)], axis=1)
            self.df['contains_april'] = self.df['text'].str.contains('april', case=False, na=False)
            self.df['april_events'] = self.df['april_events'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            self.df['pred_label'] = self.df['april_events'].apply(lambda x: 0 if x == [] else 1)
            self.df['true_label'] = self.df['contains_april'].astype(int)
            self.df.to_csv(f'{self.model}_{self.current_date}.csv')
            print(f'End Training with {self.model}')
        else:
            print(f'Model already trained. Loading the data from {self.trained_df}')
            print(f'Load Trained Model')
            self.df = pd.read_csv(self.trained_df)
            self.df = self.df.query("classes in ['business', 'sport', 'entertainment']")
            self.df['contains_april'] = self.df['text'].str.contains('april', case=False, na=False)
            self.df['april_events'] = self.df['april_events'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            self.df['pred_label'] = self.df['april_events'].apply(lambda x: 0 if x == [] else 1)
            self.df['true_label'] = self.df['contains_april'].astype(int)
            display(self.df.head(3))
            print(f'Trained Model Loaded Complete')

    def save_subcategories(self):
        print('Start Saving sub categories into new folders')

        base_folder = f'Subfolders_for_categories_{self.current_date}'
        os.makedirs(base_folder, exist_ok=True)
        
        # Categories (top-level folders)
        categories = ['business', 'sports', 'entertainment']
        
        # Step 1: Create category folders and inside them folders for unique items
        for cat in categories:
            cat_folder = os.path.join(base_folder, cat)
            os.makedirs(cat_folder, exist_ok=True)
        
            unique_subfolders = self.df[cat].dropna().unique()
            for subfolder in unique_subfolders:
                subfolder_path = os.path.join(cat_folder, subfolder)
                os.makedirs(subfolder_path, exist_ok=True)
        
        # Step 2: For each row, determine which category it belongs to (based on non-NaN value in business, sports, entertainment)
        # and write the text file inside the correct folder
        
        
        for idx, row in self.df.iterrows():
            # Find which category and subfolder this row belongs to
            for cat in categories:
                subfolder = row[cat]
                if pd.notna(subfolder):
                    # Path to write file
                    folder_path = os.path.join(base_folder, cat, subfolder)
                    os.makedirs(folder_path, exist_ok=True)  # Just in case
                    file_path = os.path.join(folder_path, f'{idx}.txt')
        
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(f"Text:\n{row['text']}\n\n")
                        f.write(f"Confidence: {row['confidence']}\n\n")
                        f.write(f"April Events:\n{row['april_events']}\n")
                    break  # Since only one category applies per row, we break after writing
        print('End Saving sub categories into new folders')
        print()
        print()
        print('===================================================================================================================================================================')
        
    def save_april(self):
        print('Start Saving April Occurences into April Folder')
        mask =  self.df['text'].str.contains(r'\bapril\b', case=False, na=False)
        filtered_dfs = self.df[mask]
        
        april_folder = f'April_{self.current_date}'
        os.makedirs(april_folder, exist_ok=True)
        
        for idx, row in filtered_dfs.iterrows():
            # Use nos if valid, else fallback to index; add idx to ensure uniqueness
            nos_val = str(row['nos']) if pd.notna(row['nos']) else 'missing'
            file_name = f"{nos_val}_{idx}.txt"
            file_path = os.path.join(april_folder, file_name)
        
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"Text:\n{row['text'] if pd.notna(row['text']) else ' '}\n\n")
                f.write(f"Business: {row['business'] if pd.notna(row['business']) else ' '}\n")
                f.write(f"Sports: {row['sports'] if pd.notna(row['sports']) else ' '}\n")
                f.write(f"Entertainment: {row['entertainment'] if pd.notna(row['entertainment']) else ' '}\n\n")
                f.write(f"Named Entities:\n{row['named_entities'] if pd.notna(row['named_entities']) else ' '}\n\n")
                f.write(f"April Events:\n{row['april_events'] if isinstance(row['april_events'], list) and row['april_events'] else (row['april_events'] if pd.notna(row['april_events']) else '')}\n")
        
        print(f"Total files written: {len(filtered_dfs)}")
        print('End Saving April Occurences into April Folder')
        print()
        print()
        print('==========================================================================================================================================================')

    def save_name_job_entities(self):
        print('Start Saving Named Entities into Name_Jobs folder')
        name_jobs_folder = f'Name_Jobs_{self.current_date}'
        os.makedirs(name_jobs_folder, exist_ok=True) 
        
        output_file = os.path.join(name_jobs_folder, 'all_named_entities.txt')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, row in self.df.iterrows():
                entities = row['named_entities'] if pd.notna(row['named_entities']) else ' '
                f.write(f"No{idx + 1}. {entities}\n\n")  # Added an extra newline here
                
        print('End Saving Named Entities into Name_Jobs folder')
        print()
        print()
        print('===========================================================================================================================================================')

    def analysis(self):
        print('Start Analysis')
        print()
        print()
        # Metrics
        accuracy = accuracy_score(self.df['true_label'], self.df['pred_label'])
        precision = precision_score(self.df['true_label'], self.df['pred_label'])
        recall = recall_score(self.df['true_label'], self.df['pred_label'])
        f1 = f1_score(self.df['true_label'], self.df['pred_label'])

        print()
        print(f'Accuracy: {accuracy:.4f}')
        print()
        print(f'Precision: {precision:.4f}')
        print()
        print(f'Recall: {recall:.4f}')
        print()
        print(f'F1 Score: {f1:.4f}')

        # Confusion Matrix
        ConfusionMatrixDisplay.from_predictions(self.df['true_label'], self.df['pred_label'], cmap='Blues')
        plt.title('Confusion Matrix')
        plt.show()
        
        print()
        print()
        print('End Analysis')
        
        print('===========================================================================================================================================================')
        
        
    def process(self):
        print(f'Start Running all the process')
        
        self.clean_data()
        self.run_model()
        self.save_subcategories()
        self.save_april()
        self.save_name_job_entities()
        self.analysis()

        print()
        print()
        print()
        print(f'Start Running all the process')
        print('===========================================================================================================================================================')

        
        

