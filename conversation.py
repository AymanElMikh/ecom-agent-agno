from agno.agent import Agent, RunResponse
import json
from instructions import Instructions
from utils import getModel
from agno.utils.log import logger

MESSAGE_SUFFIX = "\n- If any of these parameters are missing, please create a response for the user to provide them in a conversational manner and return this response under the key 'message' of the JSON."


class CategoryIdentification(Agent):
    """
    Agent responsible for identifying the product category from user input and extracting relevant requirements.
    """

    def __init__(self, api_key: str, llm_mode: str):
        """
        Initializes the CategoryIdentification agent.

        Args:
            api_key (str): API key for the language model.
            llm_mode (str): The mode of the language model (e.g., "Groq", "OpenAI").
        """
        super().__init__(
            name="Product Category Identifier",
            description="Identify the product category from the user's input and return the extra requirements in a structured format.",
            model=getModel(llm_mode, api_key),
            instructions=Instructions.IDENTIFICATION_INSTRUCTIONS,
            add_datetime_to_instructions=False,
        )

    def process_query(self, query: str) -> dict:
        """
        Processes the user's query to identify the product category and extract requirements.

        Args:
            query (str): The user's input query.

        Returns:
            dict: A dictionary containing the identified category, required keys, and a message for the user.
                  Returns a default error dictionary if JSON decoding fails or an unexpected error occurs.
        """
        enhanced_query = (
            f"{query}\n"
            "Identify the product category from the user input. "
            "Return the required information for a web search in JSON format only, including the mandatory keys. "
            "For any keys not identified from the user input, set them to `null`."
        )

        response: RunResponse = self.run(enhanced_query)

        try:
            response_json = json.loads(response.content)
            logger.info(f"Category Identification Response: {response_json}")
            return response_json

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding identification JSON: {e}")
            return {
                "category": None,
                "message": "I'm having trouble understanding your request. Could you rephrase?",
            }

        except Exception as e:
            logger.error(f"An unexpected error occurred in identification: {e}")
            return {
                "category": None,
                "message": "There was a problem 😖 please try again.",
            }


class ConversationAgent(Agent):
    """
    Agent responsible for having a conversation with the user to extract product requirements.
    """

    def __init__(self, api_key: str, llm_mode: str):
        """
        Initializes the ConversationAgent.

        Args:
            api_key (str): API key for the language model.
            llm_mode (str): The mode of the language model (e.g., "Groq", "OpenAI").
        """
        super().__init__(
            name="Conversational Product Data Extractor",
            description="Have a conversation with the user about what they are looking for to extract their requirements in a structured format.",
            model=getModel(llm_mode, api_key),
            instructions=Instructions.CONVERSATION_INSTRUCTIONS,
            add_datetime_to_instructions=True,
        )
        self.step = 0
        self.category_identifier = CategoryIdentification(api_key=api_key, llm_mode=llm_mode)
        self.requirements = None
        self.requirement_keys = None
        self.suffix = ""

    def __process_data(self, params: dict) -> dict:
        """
        Processes the extracted parameters to identify missing requirements.

        Args:
            params (dict): The extracted parameters from the user's input.

        Returns:
            dict: A dictionary containing a user message, a query suffix, and a flag indicating if there are missing parameters.
        """
        logger.info("Processing Data...")
        missing = []
        params_keys = params.keys()
        logger.info(f"Extracted Parameters: {params} | Current Requirements: {self.requirements}")

        for key in self.requirement_keys:
            try:
                if not self.requirements[key]:
                    if key in params_keys:
                        if params[key]:
                            self.requirements[key] = params[key]
                        else:
                            missing.append(key)
                    else:
                        missing.append(key)
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                for key in self.requirement_keys:
                    if self.requirements[key] is None:
                        missing.append(key)

        logger.info(f"Missing Parameters: {missing}")
        logger.info(f"Requirements after processing: {self.requirements}")

        if missing:
            missing_params = ", ".join(missing)
            return {
                "user_message": f"Please provide the following missing parameters: {missing_params}.",
                "query_suffix": f"\n- Identify the following parameters only: {missing_params} and return only these parameters in the output JSON.{MESSAGE_SUFFIX}",
                "missing": True,
            }
        else:
            return {
                "user_message": "Thank you! Fetching the products..",
                "query_suffix": "",
                "missing": False,
            }

    def __generate_response(self, data: str, load_json: bool = False) -> dict:
        """
        Generates a response to the user based on the extracted data.

        Args:
            data (str): The data received from the language model.
            load_json (bool): Whether to load the data as JSON.

        Returns:
            dict: A dictionary containing a message for the user, a flag indicating if further conversation is needed, and the extracted data.
        """
        try:
            params = data
            if load_json:
                params = json.loads(data)

            result = self.__process_data(params)
            self.suffix = result["query_suffix"]
            user_message = result["user_message"]
            have_further_conversation = result["missing"]

            if "message" not in params.keys() or not params["message"] or not have_further_conversation:
                return {
                    "message": user_message,
                    "have_further_conversation": have_further_conversation,
                    "data": self.requirements,
                }
            else:
                return {
                    "message": params["message"],
                    "have_further_conversation": have_further_conversation,
                    "data": self.requirements,
                }

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            return {
                "message": "I'm having trouble understanding your request. Could you rephrase?",
                "have_further_conversation": True,
                "data": self.requirements,
            }
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return {
                "message": "There was a problem 😖 please try again.",
                "have_further_conversation": True,
                "data": self.requirements,
            }

    def process_query(self, query: str) -> dict:
        """
        Processes the user's query to extract product requirements through conversation.

        Args:
            query (str): The user's input query.

        Returns:
            dict: A dictionary containing a message for the user, a flag indicating if further conversation is needed, and the extracted data.
        """
        if self.step == 0:
            category_data = self.category_identifier.process_query(query=query)
            self.requirements = category_data.copy()
            self.requirements.pop("message", None)
            self.requirement_keys = self.requirements.keys()
            response = self.__generate_response(data=category_data)

            if self.requirements["category"] is not None:
                self.step = 1

            return response

        if not self.suffix:
            keys = ", ".join(self.requirement_keys)
            final_query = f"{query}\n-Identify the following parameters only: {keys} and return only these keys in the output JSON.{MESSAGE_SUFFIX}"
        else:
            final_query = f"{query}\n-{self.suffix}"

        response: RunResponse = self.run(final_query)

        # Handle potential DeepSeek-specific formatting
        cleaned = response.content.split("</think>")
        if len(cleaned) > 1:
            modified = cleaned[1].replace("```json", "").replace("```", "")
        else:
            modified = cleaned[0].replace("```json", "").replace("```", "")

        return self.__generate_response(data=modified, load_json=True)

    def reset(self):
        """Resets the agent's state to start a new conversation."""
        self.step = 0
        self.requirements = None
        self.requirement_keys = None
        self.suffix = ""
