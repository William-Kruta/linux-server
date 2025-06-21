import ollama


class OllamaModel:
    def __init__(self, model_name: str, system_prompt: str, history: list = []):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.history = history
        if history == []:
            self.history.append({"role": "system", "content": self.system_prompt})

    def get_response(
        self,
        prompt: str,
        temperature: float = 0.5,
        top_p: float = 0.9,
        context_window: int = 4096,
    ) -> str:
        """
        Get a response from the ollama model.

        Parameters
        ----------
        prompt : str
            The prompt to send to the model.
        temperature : float, optional
            The temperature of the model, controls the randomness of the output, by default 0.5
        top_p : float, optional
            The top p of the model, by default 0.9
        context_window : int, optional
            The context window of the model, by default 4096
        Returns
        -------
        str
            The response from the model.
        """
        # Add system prompt if history is new.

        self.history.append({"role": "user", "content": prompt})
        model_options = {
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": context_window,
        }
        response = ollama.chat(
            model=self.model_name, messages=self.history, options=model_options
        )
        assistant_reply = response["message"]["content"]
        self.history.append({"role": "assistant", "content": assistant_reply})
        return assistant_reply
