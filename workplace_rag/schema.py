"""What we extract from each post before indexing it."""

from typing import Literal

from pydantic import BaseModel, Field

Topic = Literal[
    "work_life_balance",
    "compensation",
    "management",
    "culture",
    "career_growth",
    "layoffs_job_security",
    "interview_hiring",
    "remote_work",
    "learning_tech",
    "other",
]
Sentiment = Literal["positive", "negative", "mixed", "neutral"]
Level = Literal["intern", "junior", "mid", "senior", "manager", "unknown"]


class Opinion(BaseModel):
    topic: Topic
    sentiment: Sentiment
    company: str | None = Field(None, description="Company this opinion is about, if any")
    claim: str = Field(description="One short sentence paraphrasing what the poster says")


class Extraction(BaseModel):
    is_workplace_experience: bool = Field(
        description="True only if the post shares an experience or opinion about working somewhere"
    )
    companies: list[str] = Field(default_factory=list, description="Employers discussed, short canonical names")
    role: str | None = Field(None, description="Poster's job role, e.g. 'software engineer', 'SDE', 'data scientist'")
    city: str | None = Field(None, description="City of the job, e.g. 'Bangalore', 'Hyderabad'")
    country: str | None = None
    experience_level: Level = "unknown"
    opinions: list[Opinion] = Field(default_factory=list)
