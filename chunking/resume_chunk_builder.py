from chunking.chunk_dataclass import KnowledgeChunk
from parser.models import ParsedResume


class ResumeChunkBuilder:

    def build(self, resume: ParsedResume) -> list[KnowledgeChunk]:

        chunks: list[KnowledgeChunk] = []

        # Summary
        chunks.append(
            KnowledgeChunk(
                chunk_type="summary",
                title="Professional Summary",
                content=resume.professional_summary,
                metadata={}
            )
        )

        # Education
        for education in resume.education:
            chunks.append(
                KnowledgeChunk(
                    chunk_type="education",
                    title=education.degree,
                    content=f"""
Institution: {education.institution}
Location: {education.location}
Degree: {education.degree}
GPA: {education.gpa}
Duration: {education.start_year} - {education.end_year}
""".strip(),
                    metadata={
                        "institution": education.institution,
                        "degree": education.degree,
                    }
                )
            )

        # Experience (one chunk per highlight)
        for experience in resume.experience:

            for highlight in experience.highlights:

                chunks.append(
                    KnowledgeChunk(
                        chunk_type="experience",
                        title=f"{experience.role} @ {experience.company}",
                        content=highlight,
                        metadata={
                            "company": experience.company,
                            "role": experience.role,
                            "location": experience.location,
                            "start_date": experience.start_date,
                            "end_date": experience.end_date,
                        }
                    )
                )

        # Projects
        for project in resume.projects:

            chunks.append(
                KnowledgeChunk(
                    chunk_type="project",
                    title=project.title,
                    content=project.description,
                    metadata={
                        "technologies": project.technologies,
                        "keywords": project.keywords,
                    }
                )
            )

        # Skills
        skills = resume.skills

        chunks.append(
            KnowledgeChunk(
                chunk_type="skills",
                title="Technical Skills",
                content="\n".join([
                    f"Programming: {', '.join(skills.programming_languages)}",
                    f"AI/GenAI: {', '.join(skills.ai_genai)}",
                    f"Frameworks: {', '.join(skills.frameworks_tools)}",
                    f"Backend: {', '.join(skills.backend_infra)}",
                    f"Concepts: {', '.join(skills.concepts)}",
                ]),
                metadata={}
            )
        )

        # Achievements
        for achievement in resume.achievements:

            chunks.append(
                KnowledgeChunk(
                    chunk_type="achievement",
                    title=achievement.title,
                    content=achievement.detail,
                    metadata={}
                )
            )

        return chunks