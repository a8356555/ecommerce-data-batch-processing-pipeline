from pydantic import BaseModel

class RecommendRequest(BaseModel):
    item_id: str
    rec_num: int
