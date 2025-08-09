from typing import Annotated, Literal

from pydantic import BaseModel, Field, computed_field


class User(BaseModel):
    data_id:Annotated[int, Field(...,description="ID of the Data", examples=[1,2,3])]
    name:Annotated[str, Field(..., description="Name of the Data", examples=['Meet'])]
    age: Annotated[int, Field(...,gt=18,lt=60, description="Age of the Data", examples=[48,50,25])]
    isStudent: Annotated[bool,Field(description="is a student or not (Optional Field)")]
    gender: Annotated[Literal['male','female'],Field(..., description="Gender of the Data")]
    
    #make a new field on the spot by calculating other things
    @computed_field
    @property
    def power(self)-> str:
        if self.age > 50:
            return "Low"
        elif self.age > 30 and self.age < 50:
            return "Medium"
        elif self.age < 30:
            return "High"