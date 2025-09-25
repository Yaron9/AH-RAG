from pydantic import BaseModel, Field
from typing import List

class Entity(BaseModel):
    name: str = Field(..., description="The name of the extracted entity.")
    type: str = Field(..., description="The type of the entity (e.g., Person, Organization, Concept).")
    description: str = Field(..., description="A brief description of the entity's role in the hyperedge.")

class HypergraphExtraction(BaseModel):
    hyperedge: str = Field(..., description="A concise, one-sentence summary of the core information or event.")
    relation_type: str = Field(..., description="A short, descriptive CamelCase label for the relationship.")
    entities: List[Entity] = Field(..., description="A list of all entities involved in this hyperedge.")
    confidence_score: float = Field(..., description="Confidence in the accuracy of the extraction (1-10).")

class ExtractionResponse(BaseModel):
    extractions: List[HypergraphExtraction]
