from pydantic import BaseModel


class AskRequest(BaseModel):
	question: str


class AskResponse(BaseModel):
	id: str
	question: str
	generated_response: str
