import os
import json
from transformers import pipeline, AutoTokenizer

class EventDetector:
    def __init__(self, results_dir="results"):
        self.model_path = "raraujo/bert-finetuned-ner"
        self.results_dir = results_dir
        self.current_event = []
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.ner_pipeline = pipeline("token-classification", 
                                     model=self.model_path, 
                                     tokenizer= self.tokenizer,
                                     aggregation_strategy="simple")
        os.makedirs(self.results_dir, exist_ok=True)

    def process_message(self, message):
        ner_results = self.ner_pipeline(message['message'])
        print(f"NER Results for message '{message['message']}': {ner_results}")  # Debugging log

        if self.is_event_related(ner_results):
            self.current_event.append(message)
        else:
            if self.current_event:
                self.save_event()

    def is_event_related(self, ner_results):
        """ 
        Check if any entities in the NER results are related to events.
        
        Parameters:
            ner_results (list): A list of dictionaries representing the results of a named entity recognition (NER) analysis.
            Each dictionary should have the following keys:
                "entity_group" (str): The group to which the entity belongs, such as "DATE", "TIME", or "MEETING".
        Returns:
	        bool: True if any entities in the NER results are related to events, False otherwise.	
        """
        for entity in ner_results:
            if entity["entity_group"] in {"DATE", "TIME", "MEETING"}:
                return True
        return False

    def save_event(self):
        if not self.current_event:
            return
        event_id = len(os.listdir(self.results_dir)) + 1
        file_path = os.path.join(self.results_dir, f"event_{event_id:04d}.json")
        with open(file_path, "w") as f:
            json.dump({"lines": self.current_event}, f, indent=4)
        self.current_event = []
