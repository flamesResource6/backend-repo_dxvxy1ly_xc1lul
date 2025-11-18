from pydantic import BaseModel, Field
from typing import List, Optional

# Collections inferred from class names (lowercased):
# - Research -> "research"
# - WeeklyPlan -> "weeklyplan"

class Research(BaseModel):
    topic: str = Field(..., description="Research topic")
    summary: str = Field(..., description="Short summary of findings")

class PlanItem(BaseModel):
    day: str = Field(..., description="Day label, e.g., Monday")
    tasks: List[str] = Field(default_factory=list, description="Tasks for the day")

class WeeklyPlan(BaseModel):
    title: str = Field(..., description="Plan title")
    week_start: str = Field(..., description="ISO date of week start")
    items: List[PlanItem] = Field(default_factory=list, description="7-day plan items")
