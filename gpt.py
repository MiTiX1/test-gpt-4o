from typing import Iterable
from openai import OpenAI
from openai.types.beta.assistant import Assistant
from openai.types.beta.vector_store import VectorStore
from openai.types.beta.thread import Thread
from utils import timeit


@timeit()
def create_file_search_assistant(
    client: OpenAI,
    model: str,
    assistant_name: str,
    instructions: str
) -> Assistant:
    """
    Create a new Assistant with File Search Enabled
    """
    return client.beta.assistants.create(
        name=assistant_name,
        model=model,
        instructions=instructions,
        tools=[{"type": "file_search"}]
    )


@timeit()
def upload_file_to_vectore_store(
    client: OpenAI,
    vector_store_name: str,
    file_paths: Iterable[str]
) -> VectorStore:
    """
    Upload files and add them to a Vector Store
    """
    vector_store = client.beta.vector_stores.create(name=vector_store_name)

    file_streams = [open(path, "rb") for path in file_paths]

    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id,
        files=file_streams
    )

    print(file_batch.status)
    print(file_batch.file_counts)

    return vector_store


@timeit()
def assistant_use_vector_store(
    client: OpenAI,
    assistant: Assistant | str,
    vector_store: VectorStore | str
) -> Assistant:
    """
    Update the assistant to to use the new Vector Store
    """
    return client.beta.assistants.update(
        assistant_id=assistant.id if isinstance(assistant, Assistant) else assistant,
        tool_resources={
            "file_search": {
                "vector_store_ids": [vector_store.id if isinstance(vector_store, VectorStore) else vector_store]
            }
        }
    )


@timeit()
def create_thread(
    client: OpenAI,
    message_content: str,
    file: str = None,
) -> Thread:
    message = {
        "role": "user",
        "content": message_content,
    }

    if file:
        message_file = client.files.create(
            file=open(file, "rb"),
            purpose="assistants",
        )
        message["attachments"] = {
            "file_id": message_file.id,
            "tools": [{"type": "file_search"}]
        }

    thread = client.beta.threads.create(
        messages=[message]
    )

    print(thread.tool_resources.file_search)

    return thread
