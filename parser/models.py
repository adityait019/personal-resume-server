from pydantic import BaseModel, Field, ConfigDict, AliasChoices


class Education(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True
    )

    institution: str = Field(
        description="Name of the educational institution"
    )

    location: str | None = Field(
        default=None,
        description="Location of the educational institution"
    )

    degree: str = Field(
        description="Degree obtained"
    )

    gpa: str | None = Field(
        default=None,
        validation_alias=AliasChoices("gpa", "grade"),
        description="CGPA/GPA/Percentage"
    )

    start_year: int | str | None = Field(
        default=None,
        description="Start year"
    )

    end_year: int | str | None = Field(
        default=None,
        description="End year"
    )


class Experience(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True
    )

    company: str = Field(
        description="Company name"
    )

    location: str | None = Field(
        default=None,
        description="Company location"
    )

    role: str = Field(
        validation_alias=AliasChoices("role", "position"),
        description="Job title"
    )

    start_date: str | None = Field(
        default=None,
        description="Employment start date"
    )

    end_date: str | None = Field(
        default=None,
        description="Employment end date"
    )

    highlights: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "highlights",
            "responsibilities",
            "achievements"
        ),
        description="Key responsibilities or achievements"
    )


class Project(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True
    )

    title: str = Field(
        validation_alias=AliasChoices("title", "name"),
        description="Project title"
    )

    description: str = Field(
        description="Project description"
    )

    technologies: list[str] = Field(
        default_factory=list,
        description="Technologies used"
    )

    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords extracted from the project"
    )
    
    impact: str | None = Field(
        default=None,
        description="Impact or outcome of the project"
    )
    domain: str | None = Field(
        default=None,
        description="Domain or field of the project"
    )

class Achievement(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True
    )

    title: str = Field(
        description="Achievement title"
    )

    detail: str = Field(
        validation_alias=AliasChoices(
            "detail",
            "description"
        ),
        description="Achievement details"
    )


class Skills(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True
    )

    programming_languages: list[str] = Field(default_factory=list)

    ai_genai: list[str] = Field(default_factory=list)

    frameworks_tools: list[str] = Field(default_factory=list)

    backend_infra: list[str] = Field(default_factory=list)

    concepts: list[str] = Field(default_factory=list)


class ParsedResume(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True
    )

    name: str | None = None

    email: str | None = None

    phone: str | None = None

    linkedin: str | None = None

    github: str | None = None

    portfolio: str | None = None

    professional_summary: str = Field(
        description="Professional summary"
    )

    education: list[Education] = Field(
        default_factory=list
    )

    experience: list[Experience] = Field(
        default_factory=list
    )

    projects: list[Project] = Field(
        default_factory=list
    )

    achievements: list[Achievement] = Field(
        default_factory=list
    )

    skills: Skills