from langchain_community.chat_message_histories import KafkaChatMessageHistory


class Kafka:
    def __init__(self, chat_session_id, bootstrap_servers) -> None:
        # chat_session_id = "chat-message-history-kafka"
        # bootstrap_servers = "localhost:Plaintext Ports"
        self.history = KafkaChatMessageHistory(
            chat_session_id,
            bootstrap_servers,
        )

    def save_user_message(self, message):
        self.history.add_user_message(message)

    def save_ai_message(self, message):
        self.history.add_ai_message(message)

    def get_history(self, max_message_count=5):
        return self.history.messages_from_beginning(max_message_count)
