from services.openai import get_chat_completion
from models.models import (
    QueryResult,
    ExplainResult,
)

def get_explanation_for_query_result(query_result: QueryResult) -> str:
    """Provide the retrieved documents included in the query_result as system prompt to OpenAI to get an answer"""

    user_question = query_result.query
    relevant_docs = [result.text for result in query_result.results]
    concat_relevant_docs = "\n".join(relevant_docs)
    
    messages = [
        {
            "role": "system",
            "content": f"""
            - Answer the question to your best ability only based on these documents.
            - Omit text such as "based on these documents" or "according to the documents"
            - Answer the question directly and concisely
            - These are the documents:
            {concat_relevant_docs}
            """,
        },
        {"role": "user", "content": user_question},
    ]

    completion = get_chat_completion(
        messages,
    )

    return ExplainResult(
        query=query_result.query,
        explanation=completion,
    )
