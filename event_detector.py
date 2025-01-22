import os
import json
from transformers import pipeline, AutoTokenizer

class EventDetector:
    """
    A class to detect events from messages using a fine-tuned BERT model.

    Attributes
    ----------
    results_dir : str
        Directory to save the results.
    model_path : str
        Path to the fine-tuned BERT model.
    current_event : list
        List to store the current event.
    tokenizer : transformers.AutoTokenizer
        Tokenizer for the BERT model.
    ner_pipeline : transformers.pipelines.Pipeline
        Named Entity Recognition (NER) pipeline for token classification.

    Methods
    -------
    __init__(self, results_dir="results"):
        Initializes the EventDetector with the specified results directory.
    
    process_message(self, message):
        Processes a message to detect named entities.

    is_event_related(self, ner_results):
        Checks if any entities in the NER results are related to events.

    save_event(self):
        Saves the current event to a JSON file.
    """
     
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
        """
        Processes an incoming message to detect and handle event-related content.

        Args:
            message (dict): A dictionary containing the message data. The key 'message' 
                            should contain the text to be processed.

        Returns:
            None

        Side Effects:
            - Prints the Named Entity Recognition (NER) results for debugging purposes.
            - Appends the message to the current event if it is event-related.
            - Saves the current event if the message is not event-related and there is an ongoing event.
        """
        ner_results = self.ner_pipeline(message['message'])
        print(f"NER Results for message '{message['message']}': {ner_results}")  # Debugging log

        if self.is_event_related(ner_results):
            self.current_event.append(message)
        else:
            if self.current_event:
                self.save_event(ner_results)

    def is_event_related(self, ner_results):
        """ 
        Check if any entities in the NER results are related to events.
        
        Parameters:
            ner_results (list): A list of dictionaries representing the results of a named entity recognition (NER) analysis.
            Each dictionary should have the following keys:
                "entity_group" (str): The group to which the entity belongs, such as "DATE", "TIME", or "PLATFORM".
        Returns:
	        bool: True if any entities in the NER results are related to events, False otherwise.	
        """
        for entity in ner_results:
            if entity["entity_group"] in {"DATE", "TIME", "PLATFORM"}:
                return True
        return False

    def save_event(self, ner_results):
        """
        Save the current event and NER results to a JSON file in the results directory.

        This function generates a unique event ID based on the number of files
        already present in the results directory. It then saves the current event
        and NER results as a JSON file with the name format 'event_<event_id>.json'.
        After saving, it clears the current event list.

        Parameters
        ----------
        ner_results : list
            A list of dictionaries representing the results of a named entity recognition (NER) analysis.
        """
        # If there are no events to save, return immediately
        if not self.current_event:
            return

        # Generate a unique event ID based on the number of files in the results directory
        event_id = len(os.listdir(self.results_dir)) + 1

        # Create the file path for the new event file
        file_path = os.path.join(self.results_dir, f"event_{event_id:04d}.json")

        # Save the current event and NER results to the file in JSON format
        with open(file_path, "w") as f:
            json.dump({"lines": self.current_event, "ner_results": ner_results}, f, indent=4)

        # Clear the current event list
        self.current_event = []
