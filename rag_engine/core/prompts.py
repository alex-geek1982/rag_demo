"""
Knowledge Graph Extraction Prompts (Inspired by LightRAG)

This module contains prompts for entity and relationship extraction using LLM,
following LightRAG's pattern with comprehensive system and user prompts.

Reference: https://github.com/HKUDS/LightRAG/blob/main/lightrag/prompt.py
"""

# Delimiters for structured extraction
TUPLE_DELIMITER = "<|#|>"
COMPLETION_DELIMITER = "<|COMPLETE|>"

ENTITY_EXTRACTION_SYSTEM_PROMPT = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
        *   `entity_type`: Categorize the entity using one of the following types: `{entity_types}`. If none of the provided entity types apply, do not add new entity type and classify it as `Other`.
        *   `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
    *   **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
        *   **Example:** For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Project X," "Bob collaborated with Project X," and "Carol collaborated with Project X," or "Alice collaborated with Bob," based on the most reasonable binary interpretations.
    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `relationship_keywords`: One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship. Multiple keywords within this field must be separated by a comma `,`. **DO NOT use `{tuple_delimiter}` for separating multiple keywords within this field.**
        *   `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities, providing a clear rationale for their connection.
    *   **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

7.  **Language & Proper Nouns:** 
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

---Examples---
{examples}
"""

ENTITY_CONTINUE_EXTRACTION_SYSTEM_PROMPT = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: The name of the entity. If the entity name is case-insensitive, capitalize the first letter of each significant word (title case). Ensure **consistent naming** across the entire extraction process.
        *   `entity_type`: Categorize the entity using one of the following types: `{entity_types}`. If none of the provided entity types apply, do not add new entity type and classify it as `Other`.
        *   `entity_description`: Provide a concise yet comprehensive description of the entity's attributes and activities, based *solely* on the information present in the input text.
    *   **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Identification:** Identify direct, clearly stated, and meaningful relationships between previously extracted entities.
    *   **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
        *   **Example:** For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Project X," "Bob collaborated with Project X," and "Carol collaborated with Project X," or "Alice collaborated with Bob," based on the most reasonable binary interpretations.
    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `relationship_keywords`: One or more high-level keywords summarizing the overarching nature, concepts, or themes of the relationship. Multiple keywords within this field must be separated by a comma `,`. **DO NOT use `{tuple_delimiter}` for separating multiple keywords within this field.**
        *   `relationship_description`: A concise explanation of the nature of the relationship between the source and target entities, providing a clear rationale for their connection.
    *   **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

7.  **Language & Proper Nouns:** 
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

---Examples---
{examples}
"""

ENTITY_EXTRACTION_USER_PROMPT = """---Task---
Extract entities and relationships from the input text in Data to be Processed below.

---Instructions---
1.  **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.
2.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
3.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant entities and relationships have been extracted and presented.
4.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

---Data to be Processed---
<Entity_types>
[{entity_types}]

<Input Text>
```
{input_text}
```

<Output>
"""


# Continuation extraction user prompt (gleaning)
ENTITY_CONTINUE_EXTRACTION_USER_PROMPT = """---Task---
Review the input text and the previously extracted entities and relationships.
Identify any entities, relationships, or corrections that were missed or incomplete in the previous extraction.

---Instructions---

1. **Comparison Focus:** Compare your analysis against the previous extraction provided below.

2. **Identify Gaps:**
   - Implicit entities (concepts, attributes, characteristics) not explicitly named
   - Indirect or contextual relationships
   - Entities truncated or incomplete in previous result
   - Relationships with missing or vague descriptions

3. **Output ONLY Missing Items:** Do not repeat items that were correctly extracted before.

4. **Completion Signal:** End with `{completion_delimiter}` after all missing items are listed.

---Previous Extraction Result---
{previous_extraction}

---Input Text---
```
{input_text}
```

---Output (Only New/Corrected Items)---
"""

# Summarization prompt
ENTITY_DESCRIPTION_SUMMARIZATION_SYSTEM_PROMPT = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Synthesize a list of descriptions of a given entity or relationship into a single, comprehensive, and cohesive summary.

---Instructions---

1. **Comprehensiveness:** The summary must integrate ALL key information from every provided description. Do not omit any important facts, attributes, or details.

2. **Context:** 
   - Ensure the summary is written from an objective, third-person perspective.
   - Explicitly mention the name of the entity or relationship at the beginning.
   - Preserve specific details, dates, numbers, and qualitative attributes.

3. **Conflict Handling:**
   - If conflicting or inconsistent descriptions arise:
     - Attempt to reconcile them (e.g., different time periods, different perspectives)
     - If they represent distinct aspects, include both with a connector like "historically" or "in different contexts"
     - If genuinely contradictory, note both viewpoints with qualifying language
   - Do NOT discard information due to conflicts; synthesize it intelligently.

4. **Organization:**
   - Group related information logically
   - Use topic sentences to guide readers
   - Maintain chronological or hierarchical order where appropriate

5. **Length Constraint:** 
   - The summary must be comprehensive while remaining reasonably concise
   - Typically 2-5 sentences for entity descriptions
   - Typically 1-3 sentences for relationship descriptions

6. **Language & Proper Nouns:** 
   - The entire output must be in the specified language
   - Retain proper nouns in their original language
   - Preserve technical terms and domain-specific vocabulary

7. **Output Format:** Provide ONLY the synthesized summary as plain text. No additional formatting, prefix, or explanation.
"""

# Summarization user prompt
ENTITY_DESCRIPTION_SUMMARIZATION_USER_PROMPT = """---Task---
Synthesize the following {description_type} descriptions into a single, comprehensive summary.

Entity/Relationship Name: {name}

---Descriptions to Synthesize---
{descriptions_text}

---Instructions---
1. Integrate all key information from the descriptions above.
2. Create a unified, coherent summary that captures the essence of all provided descriptions.
3. Do not introduce information not present in the source descriptions.
4. Output ONLY the synthesized summary (no prefix, no explanation).

---Synthesis---
"""

# Example for entity extraction (optional for few-shot prompting)
ENTITY_EXTRACTION_EXAMPLES = """Example 1:
Entity Types: ["Person", "Organization", "Location", "Technology", "Event", "Product"]
Input Text:
```
Apple Inc., headquartered in Cupertino, California, announced a groundbreaking partnership with OpenAI at the WWDC conference. CEO Tim Cook revealed the new AI-powered features integrated into iOS 18, which utilizes advanced machine learning algorithms originally developed by OpenAI researchers.
```

Output:
entity<|#|>Apple Inc.<|#|>Organization<|#|>Apple Inc. is a multinational technology company headquartered in Cupertino, California, known for developing consumer electronics, software, and services.
entity<|#|>Cupertino<|#|>Location<|#|>Cupertino is a city in California where Apple Inc. is headquartered.
entity<|#|>California<|#|>Location<|#|>California is a state in the United States where Cupertino is located.
entity<|#|>OpenAI<|#|>Organization<|#|>OpenAI is an AI research organization that partnered with Apple Inc. and provided technology for iOS 18 features.
entity<|#|>Tim Cook<|#|>Person<|#|>Tim Cook is the CEO of Apple Inc. who announced the partnership with OpenAI.
entity<|#|>WWDC<|#|>Event<|#|>WWDC is a conference where Apple Inc. announced its partnership with OpenAI.
entity<|#|>iOS 18<|#|>Product<|#|>iOS 18 is a software product developed by Apple Inc. that includes AI-powered features.
entity<|#|>Machine Learning<|#|>Technology<|#|>Machine learning is a technology area that forms the basis of the AI features in iOS 18.
relation<|#|>Apple Inc.<|#|>OpenAI<|#|>partnership, collaboration<|#|>Apple Inc. announced a groundbreaking partnership with OpenAI for AI integration.
relation<|#|>Apple Inc.<|#|>Cupertino<|#|>headquarters, location<|#|>Apple Inc. is headquartered in Cupertino, California.
relation<|#|>Tim Cook<|#|>Apple Inc.<|#|>CEO, leadership<|#|>Tim Cook serves as the CEO of Apple Inc.
relation<|#|>iOS 18<|#|>OpenAI<|#|>integration, AI features<|#|>iOS 18 integrates technology from OpenAI for advanced AI capabilities.
relation<|#|>iOS 18<|#|>Machine Learning<|#|>utilizes, foundation<|#|>iOS 18 utilizes machine learning algorithms for its advanced features.
<|COMPLETE|>"""

# Default prompts dictionary
DEFAULT_PROMPTS = {
    "entity_extraction_system": ENTITY_EXTRACTION_SYSTEM_PROMPT,
    "entity_extraction_user": ENTITY_EXTRACTION_USER_PROMPT,
    "entity_continue_extraction_system": ENTITY_CONTINUE_EXTRACTION_SYSTEM_PROMPT,
    "entity_summarization_system": ENTITY_DESCRIPTION_SUMMARIZATION_SYSTEM_PROMPT,
    "tuple_delimiter": TUPLE_DELIMITER,
    "completion_delimiter": COMPLETION_DELIMITER,
}

def get_relationship_keywords_extraction_prompt(language: str = "English") -> str:
    """
    Get prompt for extracting keywords from relationships.
    
    Args:
        language: Language for extraction
    
    Returns:
        Formatted prompt
    """
    return f"""Extract high-level keywords that summarize the nature and essence of the relationship described below. 
    
The keywords should capture the core concept or theme of the relationship, not just surface details.
Output format: keyword1, keyword2, keyword3

Relationship: {{relationship}}

Keywords (in {language}):"""


def get_continue_extraction_prompt(input_text: str, previous_extraction: str,
                                   tuple_delimiter: str = TUPLE_DELIMITER,
                                   completion_delimiter: str = COMPLETION_DELIMITER,
                                   language: str = "English") -> str:
    """
    Get formatted continue extraction (gleaning) user prompt (LightRAG style).
    
    Args:
        input_text: The text to re-analyze
        previous_extraction: The previous extraction result
        tuple_delimiter: Delimiter for tuple fields
        completion_delimiter: Completion signal delimiter
        language: Language for extraction
    
    Returns:
        Formatted user prompt
    """
    prompt = f"""---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Strict Adherence to System Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system instructions.
2.  **Focus on Corrections/Additions:**
    *   **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
    *   If an entity or relationship was **missed** in the last task, extract and output it now according to the system format.
    *   If an entity or relationship was **truncated, had missing fields, or was otherwise incorrectly formatted** in the last task, re-output the *corrected and complete* version in the specified format.
3.  **Output Format - Entities:** Output a total of 4 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
4.  **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
5.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
6.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant missing or corrected entities and relationships have been extracted and presented.
7.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

---Previous Extraction Result---
{previous_extraction}

---Input Text---
```
{input_text}
```

<Output>
"""
    
    return prompt


def get_continue_extraction_system_prompt(tuple_delimiter: str = TUPLE_DELIMITER,
                                         completion_delimiter: str = COMPLETION_DELIMITER,
                                         entity_types: list = None,
                                         language: str = "English",
                                         examples: str = "") -> str:
    """
    Get formatted continue extraction system prompt.
    
    Args:
        tuple_delimiter: Delimiter for tuple fields
        completion_delimiter: Completion signal delimiter
        entity_types: List of entity types to extract. Uses default if not provided.
        language: Language for extraction
        examples: Example extractions. Uses default if not provided.
    
    Returns:
        Formatted system prompt
    """
    if entity_types is None:
        entity_types = [
            "Person", "Creature", "Organization", "Location", "Event",
            "Concept", "Method", "Content", "Data", "Artifact", "NaturalObject"
        ]
    
    entity_types_str = ", ".join(entity_types)
    
    prompt = ENTITY_CONTINUE_EXTRACTION_SYSTEM_PROMPT.format(
        tuple_delimiter=tuple_delimiter,
        completion_delimiter=completion_delimiter,
        entity_types=entity_types_str,
        language=language,
        examples=examples
    )
    return prompt


def get_summarization_prompt(descriptions: list, entity_name: str, 
                            description_type: str = "Entity",
                            language: str = "English") -> str:
    """
    Get formatted description summarization prompt.
    
    Args:
        descriptions: List of descriptions to summarize
        entity_name: Name of entity or relationship
        description_type: "Entity" or "Relationship"
        language: Language for summarization
    
    Returns:
        Formatted user prompt
    """
    # Join descriptions with proper formatting
    descriptions_text = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(descriptions)])
    
    prompt = ENTITY_DESCRIPTION_SUMMARIZATION_USER_PROMPT.format(
        description_type=description_type,
        name=entity_name,
        descriptions_text=descriptions_text,
    )
    
    return prompt


def get_system_prompt(entity_types: list = None, language: str = "English", examples: str = "") -> str:
    """
    Get entity extraction system prompt (LightRAG style).
    
    Args:
        entity_types: List of entity types for extraction
        language: Language for extraction
        examples: Example extractions for few-shot prompting
    
    Returns:
        Formatted system prompt
    """
    if entity_types is None:
        entity_types = [
            "Person", "Creature", "Organization", "Location", "Event",
            "Concept", "Method", "Content", "Data", "Artifact", "NaturalObject"
        ]
    
    # Format the prompt with entity types and other parameters
    entity_types_str = ", ".join(entity_types)
    return ENTITY_EXTRACTION_SYSTEM_PROMPT.format(
        entity_types=entity_types_str,
        language=language,
        tuple_delimiter=TUPLE_DELIMITER,
        completion_delimiter=COMPLETION_DELIMITER,
        examples=examples
    )


def get_summarization_system_prompt(language: str = "English") -> str:
    """
    Get description summarization system prompt.
    
    Args:
        language: Language for summarization
    
    Returns:
        Formatted system prompt for summarization
    """
    return ENTITY_DESCRIPTION_SUMMARIZATION_SYSTEM_PROMPT


def get_user_prompt(input_text: str, entity_types: list = None,
                   tuple_delimiter: str = TUPLE_DELIMITER,
                   completion_delimiter: str = COMPLETION_DELIMITER,
                   language: str = "English") -> str:
    """
    Get formatted entity extraction user prompt (LightRAG style).
    
    Args:
        input_text: The text to extract entities from
        entity_types: List of entity types (for reference in prompt)
        tuple_delimiter: Delimiter for tuple fields
        completion_delimiter: Completion signal delimiter
        language: Language for extraction
    
    Returns:
        Formatted user prompt
    """
    if entity_types is None:
        entity_types = [
            "Person", "Creature", "Organization", "Location", "Event",
            "Concept", "Method", "Content", "Data", "Artifact", "NaturalObject"
        ]
    
    entity_types_str = ", ".join(entity_types)
    
    prompt = ENTITY_EXTRACTION_USER_PROMPT.format(
        input_text=input_text,
        entity_types=entity_types_str,
        tuple_delimiter=tuple_delimiter,
        completion_delimiter=completion_delimiter,
        language=language
    )
    
    return prompt

