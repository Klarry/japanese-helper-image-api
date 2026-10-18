# HTTP API reference

Generated from the FastAPI app's OpenAPI schema by `python -m app.export_api_docs`.
Do not edit by hand - change the code and regenerate.

44 endpoints. Interactive version: `/docs` on a running server.

## /agent

### POST /agent/chat

Agent Chat

Request body: `AgentChatRequest`
- `message`: string (required)
- `strategy`: ContextStrategy | null (optional)
- `compression_enabled`: boolean | null (optional)

Response 200: `AgentChatResponse`
- `response`: string (required)
- `usage`: AgentTokenUsage (required)
- `compression`: AgentCompressionStatus (required)
- `strategy`: string (optional)
- `tool_calls`: list[AgentToolCall] (optional)

### GET /agent/history

Agent History

Response 200: `AgentHistoryResponse`
- `messages`: list[AgentHistoryMessage] (required)
- `summary`: string (optional)

### DELETE /agent/history

Clear Agent History

Response 200: `AgentHistoryResponse`
- `messages`: list[AgentHistoryMessage] (required)
- `summary`: string (optional)

### PUT /agent/strategy

Choose how much of the conversation the next requests will send.

Request body: `AgentStrategyRequest`
- `strategy`: ContextStrategy (required)

Response 200: `AgentStrategyResponse`
- `strategy`: string (required)

### GET /agent/context

The context the next request would send, plus the branch and checkpoints it would send it from.

Response 200: `AgentContextResponse`
- `strategy`: string (required)
- `branch`: string (required)
- `branches`: list[string] (required)
- `checkpoints`: list[string] (required)
- `facts`: object (optional)
- `messages`: list[AgentHistoryMessage] (optional)
- `context`: string (optional)

### POST /agent/checkpoint

Capture the current branch so branches can be forked from this point.

Request body: `AgentCheckpointRequest | null`

Response 200: `AgentCheckpointResponse`
- `name`: string (required)
- `branch`: string (required)
- `messages`: integer (required)

### PUT /agent/branch

Talk on another branch from now on.

Request body: `AgentBranchSwitchRequest`
- `name`: string (required)

Response 200: `AgentBranchResponse`
- `branch`: string (required)
- `branches`: list[string] (required)

### POST /agent/branch

Fork a branch from a checkpoint. Does not switch to it.

Request body: `AgentBranchRequest`
- `name`: string (required)
- `checkpoint`: string (required)

Response 200: `AgentBranchResponse`
- `branch`: string (required)
- `branches`: list[string] (required)

### GET /agent/memory

All three memory layers, each under its own key: the current conversation, the current task, and what is remembered about the learner across conversations.

Response 200: `AgentMemoryResponse`
- `short_term`: AgentShortTermMemory (optional)
- `working`: AgentWorkingMemory (optional)
- `long_term`: AgentLongTermMemory (optional)

### PUT /agent/memory/short_term

Replace the current conversation.

Request body: `AgentShortTermMemoryRequest`
- `messages`: list[AgentHistoryMessage] (optional)

Response 200: `AgentMemoryResponse`
- `short_term`: AgentShortTermMemory (optional)
- `working`: AgentWorkingMemory (optional)
- `long_term`: AgentLongTermMemory (optional)

### PUT /agent/memory/working

Update the current task's goals, requirements, constraints and decisions. Fields left out keep their current value.

Request body: `AgentWorkingMemoryRequest`
- `goals`: list[string] | null (optional)
- `requirements`: list[string] | null (optional)
- `constraints`: list[string] | null (optional)
- `decisions`: list[string] | null (optional)

Response 200: `AgentMemoryResponse`
- `short_term`: AgentShortTermMemory (optional)
- `working`: AgentWorkingMemory (optional)
- `long_term`: AgentLongTermMemory (optional)

### PUT /agent/memory/long_term

Update what is remembered about the learner between conversations.

Request body: `AgentLongTermMemoryRequest`
- `profile`: object | null (optional)
- `preferences`: list[string] | null (optional)
- `decisions`: list[string] | null (optional)
- `knowledge`: list[string] | null (optional)

Response 200: `AgentMemoryResponse`
- `short_term`: AgentShortTermMemory (optional)
- `working`: AgentWorkingMemory (optional)
- `long_term`: AgentLongTermMemory (optional)

### DELETE /agent/memory/{layer}

Empty one layer - short_term, working or long_term - and leave the other two exactly as they are.

Parameters:
- `layer` (path): MemoryLayer

Response 200: `AgentMemoryResponse`
- `short_term`: AgentShortTermMemory (optional)
- `working`: AgentWorkingMemory (optional)
- `long_term`: AgentLongTermMemory (optional)

### GET /agent/profile

How the learner wants to be answered. Read on every chat request, so the preferences never have to be repeated in a message.

Response 200: `AgentUserProfile`
- `preferred_language`: string (optional)
- `japanese_level`: string (optional)
- `explanation_style`: string (optional)
- `answer_format`: string (optional)
- `translation_language`: string (optional)
- `preferences`: list[string] (optional)

### PUT /agent/profile

Update the profile. Fields left out keep their current value.

Request body: `AgentUserProfileRequest`
- `preferred_language`: string | null (optional)
- `japanese_level`: string | null (optional)
- `explanation_style`: string | null (optional)
- `answer_format`: string | null (optional)
- `translation_language`: string | null (optional)
- `preferences`: list[string] | null (optional)

Response 200: `AgentUserProfile`
- `preferred_language`: string (optional)
- `japanese_level`: string (optional)
- `explanation_style`: string (optional)
- `answer_format`: string (optional)
- `translation_language`: string (optional)
- `preferences`: list[string] (optional)

### DELETE /agent/profile

Unset every setting. Leaves all three memory layers untouched.

Response 200: `AgentUserProfile`
- `preferred_language`: string (optional)
- `japanese_level`: string (optional)
- `explanation_style`: string (optional)
- `answer_format`: string (optional)
- `translation_language`: string (optional)
- `preferences`: list[string] (optional)

### GET /agent/invariants

Every rule the agent is bound by. Read on every chat request, so a rule added here applies to the next message.

Response 200: `AgentInvariantsResponse`
- `invariants`: list[AgentInvariant] (optional)

### POST /agent/invariants

Add a rule. The id is generated from the category.

Request body: `AgentInvariantRequest`
- `category`: InvariantCategory (required)
- `rule`: string (required)

Response 200: `AgentInvariantsResponse`
- `invariants`: list[AgentInvariant] (optional)

### PUT /agent/invariants/{invariant_id}

Add or change the rule with this id, keeping its place in the list.

Parameters:
- `invariant_id` (path): string

Request body: `AgentInvariantRequest`
- `category`: InvariantCategory (required)
- `rule`: string (required)

Response 200: `AgentInvariantsResponse`
- `invariants`: list[AgentInvariant] (optional)

### DELETE /agent/invariants/{invariant_id}

Remove one rule. Unknown ids are reported rather than ignored.

Parameters:
- `invariant_id` (path): string

Response 200: `AgentInvariantsResponse`
- `invariants`: list[AgentInvariant] (optional)

### GET /agent/digest

What the periodic digest task has collected so far: how many runs, when the last one was, how many words, and a line of summary. Reading it changes nothing - the runs happen on their own schedule.

Response 200: `AgentDigestResponse`
- `found`: boolean (optional)
- `summary`: string (optional)
- `task_id`: string (optional)
- `query`: string (optional)
- `interval_seconds`: integer (optional)
- `active`: boolean (optional)
- `runs`: integer (optional)
- `failed_runs`: integer (optional)
- `last_run`: string (optional)
- `next_run`: string (optional)
- `items_collected`: integer (optional)
- `unique_words`: integer (optional)
- `levels`: object (optional)
- `latest_items`: list[object] (optional)
- `last_error`: string (optional)

### POST /agent/rag

Answer a question about the project's own documents (Days 22-23).

Request body: `AgentRagRequest`
- `question`: string (required)
- `use_rag`: boolean (optional)
- `top_k`: integer (optional)
- `mode`: string | null (optional)
- `retrieval_top_k`: integer | null (optional)
- `similarity_threshold`: number | null (optional)
- `final_top_k`: integer | null (optional)
- `query_rewrite`: boolean | null (optional)

Response 200: `AgentRagResponse`
- `answer`: string (required)
- `rag_enabled`: boolean (optional)
- `mode`: string (optional)
- `sources`: list[string] (optional)
- `retrieved_chunks`: list[AgentRagChunk] (optional)
- `top_k`: integer (optional)
- `embedding_model`: string (optional)
- `retrieval_seconds`: number (optional)
- `llm_seconds`: number (optional)
- `debug`: AgentRagDebug | null (optional)
- `rag_status`: string (optional)
- `confidence`: string (optional)
- `citation_support`: string (optional)
- `cited_sources`: list[AgentRagSource] (optional)
- `citations`: list[AgentRagCitation] (optional)

### POST /agent/mini-chat

One turn of the mini chat (Day 25).

Request body: `AgentMiniChatRequest`
- `message`: string (required)

Response 200: `AgentMiniChatResponse`
- `answer`: string (required)
- `rag_status`: string (optional)
- `confidence`: string (optional)
- `sources`: list[AgentRagSource] (optional)
- `citations`: list[AgentRagCitation] (optional)
- `task_memory`: AgentTaskMemory (optional)
- `memory_changes`: list[string] (optional)
- `retrieved_count`: integer (optional)
- `filtered_count`: integer (optional)
- `final_count`: integer (optional)
- `best_relevance`: number (optional)
- `answer_threshold`: number (optional)
- `history_length`: integer (optional)
- `seconds`: number (optional)

### GET /agent/documents

What the local document index holds: how many documents went in, how many chunks each strategy made of them, and what embedded them. Built by `python -m app.index_documents`; this only reads the report.

Response 200: `AgentDocumentIndexResponse`
- `found`: boolean (optional)
- `documents`: integer (optional)
- `total_characters`: integer (optional)
- `fixed_chunks`: integer (optional)
- `structural_chunks`: integer (optional)
- `embedding_model`: string (optional)
- `embedding_dimension`: integer (optional)
- `built_at`: string (optional)

### GET /agent/task

Where the task in progress has got to, and which stages it may move to from here.

Response 200: `AgentTaskStateResponse`
- `task_stage`: string (optional)
- `current_step`: string (optional)
- `expected_action`: string (optional)
- `allowed_next`: list[string] (optional)
- `plan`: string (optional)
- `validation_passed`: boolean (optional)
- `validation_note`: string (optional)
- `next_requirement`: string (optional)
- `blocked`: AgentTaskTransitionError | null (optional)

### DELETE /agent/task

End the task. The conversation, the memory layers and the profile are left untouched.

Response 200: `AgentTaskStateResponse`
- `task_stage`: string (optional)
- `current_step`: string (optional)
- `expected_action`: string (optional)
- `allowed_next`: list[string] (optional)
- `plan`: string (optional)
- `validation_passed`: boolean (optional)
- `validation_note`: string (optional)
- `next_requirement`: string (optional)
- `blocked`: AgentTaskTransitionError | null (optional)

### POST /agent/task/transition

Move the task to a named stage.

Request body: `AgentTaskTransitionRequest`
- `task_stage`: TaskStage (required)
- `current_step`: string (optional)
- `expected_action`: string (optional)

Response 200: `AgentTaskStateResponse`
- `task_stage`: string (optional)
- `current_step`: string (optional)
- `expected_action`: string (optional)
- `allowed_next`: list[string] (optional)
- `plan`: string (optional)
- `validation_passed`: boolean (optional)
- `validation_note`: string (optional)
- `next_requirement`: string (optional)
- `blocked`: AgentTaskTransitionError | null (optional)

### POST /agent/task/plan

Approve the plan the task will be executed by - the condition for leaving planning. 409 unless the task is planning.

Request body: `AgentTaskPlanRequest`
- `plan`: string (required)

Response 200: `AgentTaskStateResponse`
- `task_stage`: string (optional)
- `current_step`: string (optional)
- `expected_action`: string (optional)
- `allowed_next`: list[string] (optional)
- `plan`: string (optional)
- `validation_passed`: boolean (optional)
- `validation_note`: string (optional)
- `next_requirement`: string (optional)
- `blocked`: AgentTaskTransitionError | null (optional)

### POST /agent/task/validation

Record how validation went - the condition for reaching done. 409 unless the task is validating.

Request body: `AgentTaskValidationRequest`
- `passed`: boolean (required)
- `notes`: string (optional)

Response 200: `AgentTaskStateResponse`
- `task_stage`: string (optional)
- `current_step`: string (optional)
- `expected_action`: string (optional)
- `allowed_next`: list[string] (optional)
- `plan`: string (optional)
- `validation_passed`: boolean (optional)
- `validation_note`: string (optional)
- `next_requirement`: string (optional)
- `blocked`: AgentTaskTransitionError | null (optional)

### GET /agent/usage

Token usage of every recorded chat request, tagged with the strategy that produced it - the raw material for comparing them.

Response 200: `AgentUsageResponse`
- `entries`: list[AgentUsageEntry] (required)

## /code-review

### POST /code-review

Review a pull request's diff with RAG context from the project index. Returns validated findings (bugs, security, architecture, tests, ...), notes, recommendations and the limitations of the review.

Request body: `CodeReviewRequest`
- `repository`: string (required)
- `pull_request_number`: integer (required)
- `base_branch`: string (required)
- `head_branch`: string (required)
- `title`: string (optional)
- `diff`: string (required)
- `changed_files`: list[ChangedFile] (optional)
- `provider`: string (optional)

Response 200: `CodeReviewResponse`
- `request_id`: string (required)
- `repository`: string (required)
- `pull_request_number`: integer (required)
- `base_branch`: string (required)
- `head_branch`: string (required)
- `summary`: string (required)
- `findings`: list[CodeReviewFinding] (required)
- `architecture_notes`: list[string] (optional)
- `recommendations`: list[string] (optional)
- `limitations`: list[string] (optional)
- `provider`: string (required)
- `model`: string (required)
- `files`: list[object] (optional)
- `context`: object (optional)
- `validation`: object (optional)
- `generation`: object (optional)
- `timing`: object (optional)

## /description

### POST /description

Description

Request body: `DescriptionRequest`
- `meaning`: string (required)

Response 200: `DescriptionResponse`
- `uncontrolled`: string (required)
- `controlled`: string (required)

## /dev

### POST /dev/chat

Answer a /help command from the project's documentation (RAG) and its live Git state (MCP get_git_branch).

Request body: `DevChatRequest`
- `message`: string (required)
- `provider`: string (optional)

Response 200: `DevChatResponse`
- `command`: string (required)
- `question`: string (required)
- `answer`: string (required)
- `sources`: list[RagChatSource] (optional)
- `citations`: list[RagChatCitation] (optional)
- `rag_status`: string (required)
- `confidence`: string (optional)
- `provider`: string (required)
- `model`: string | null (optional)
- `tools_used`: list[DevChatToolUse] (optional)
- `timing`: object (optional)
- `best_relevance`: number (optional)
- `retrieval`: object | null (optional)

## /image-search

### POST /image-search

Image Search

Request body: `ImageSearchRequest`
- `query`: string (required)

## /kanji-word-set

### POST /kanji-word-set

Kanji Word Set

Request body: `KanjiWordSetRequest`
- `kanji`: string (required)
- `experimentType`: ExperimentType (required)

Response 200: `KanjiWordSetResponse`
- `prompt`: string (required)
- `words`: list[string] (required)
- `cost`: integer (required)
- `value`: integer (required)

## /local-chat

### POST /local-chat

The app's chat (Day 27), answered by the model on this machine - and since Day 30, by a service other machines on the network can use.

Request body: `LocalChatRequest`
- `message`: string (required)
- `conversation_id`: string | null (optional)

## /local-llm

### POST /local-llm

Local Llm

Request body: `LocalLlmRequest`
- `prompt`: string (required)

Response 200: `LocalLlmResponse`
- `model`: string (required)
- `response`: string (required)

### GET /local-llm/health

200 with the verdict in the body - or, with ``?strict=true``, 503 when the model cannot answer.

Parameters:
- `strict` (query): boolean

## /model-comparison

### POST /model-comparison

Model Comparison

Request body: `ModelComparisonRequest`
- `kanji`: string (required)
- `prompt`: string (required)
- `models`: list[string] (required)

Response 200: `ModelComparisonResponse`
- `prompt`: string (required)
- `results`: list[ModelComparisonResult] (required)

## /project-files

### POST /project-files/task

Carry out a task on the project's files: find usages of a component or API and explain the flow, or compare documentation with the code and propose a patch. The assistant chooses the files.

Request body: `ProjectTaskRequest`
- `task`: string (required)
- `provider`: string (optional)

Response 200: `ProjectTaskResponse`
- `task`: string (required)
- `status`: string (required)
- `workflow`: string (required)
- `subject`: string (required)
- `summary`: string (required)
- `provider`: string (required)
- `model`: string (required)
- `run_id`: string (required)
- `plan`: object (optional)
- `files_inspected`: list[FileInspected] (optional)
- `artifacts`: list[TaskArtifact] (optional)
- `diffs`: list[ProposedDiff] (optional)
- `tool_calls`: list[TaskToolCall] (optional)
- `model_calls`: list[TaskModelCall] (optional)
- `findings`: object (optional)
- `errors`: list[TaskError] (optional)
- `notes`: list[string] (optional)
- `limits`: object (optional)
- `timing`: object (optional)
- `run_record`: string (optional)

### POST /project-files/apply

Show the change a saved proposal makes; with confirm, make it - only if the file still has the sha256 the proposal was made against.

Request body: `ApplyRequest`
- `path`: string (required)
- `proposal`: string (required)
- `expected_sha256`: string (required)
- `confirm`: boolean (optional)

Response 200: `ApplyResponse`
- `path`: string (required)
- `proposal`: string (required)
- `applied`: boolean (required)
- `diff`: string (required)
- `added`: integer (required)
- `removed`: integer (required)
- `base_sha256`: string | null (optional)
- `sha256`: string | null (optional)
- `message`: string (required)

## /rag

### POST /rag/chat

Rag Chat

Request body: `RagChatRequest`
- `message`: string (required)
- `provider`: string (optional)

Response 200: `RagChatResponse`
- `answer`: string (required)
- `sources`: list[RagChatSource] (optional)
- `citations`: list[RagChatCitation] (optional)
- `confidence`: string (required)
- `rag_status`: string (required)
- `provider`: string (required)
- `model`: string (required)
- `retrieved_count`: integer (optional)
- `final_count`: integer (optional)
- `best_relevance`: number (optional)
- `answer_threshold`: number (optional)
- `timing`: RagChatTiming (required)

## /support

### POST /support/chat

Answer a user's question about the product, with their ticket (via MCP) and the help centre (via RAG) as context.

Request body: `SupportChatRequest`
- `message`: string (required)
- `ticket_id`: string (required)
- `provider`: string (optional)

Response 200: `SupportChatResponse`
- `request_id`: string (required)
- `answer`: string (required)
- `ticket_id`: string (required)
- `status`: string (required)
- `sources`: list[RagChatSource] (optional)
- `citations`: list[RagChatCitation] (optional)
- `rejected_citations`: integer (optional)
- `ticket`: SupportTicketContext (required)
- `ticket_facts_used`: list[string] (optional)
- `tools_used`: list[SupportToolUse] (optional)
- `provider`: string (required)
- `model`: string (required)
- `retrieval`: object (optional)
- `timing`: object (optional)

## /temperature-description

### POST /temperature-description

Temperature Description

Request body: `TemperatureDescriptionRequest`
- `kanji`: string (required)
- `temperature`: number (required)

Response 200: `TemperatureDescriptionResponse`
- `sentence`: string (required)
- `translation`: string (required)
- `temperature`: number (required)
