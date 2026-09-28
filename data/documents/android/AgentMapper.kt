package com.japanesehelper.data.mapper

import com.google.gson.Gson
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonObject
import com.japanesehelper.data.remote.dto.AgentChatResponseDto
import com.japanesehelper.data.remote.dto.AgentToolCallDto
import com.japanesehelper.data.remote.dto.AgentContextResponseDto
import com.japanesehelper.data.remote.dto.AgentDigestDto
import com.japanesehelper.data.remote.dto.AgentHistoryMessageDto
import com.japanesehelper.data.remote.dto.AgentHistoryResponseDto
import com.japanesehelper.data.remote.dto.AgentInvariantDto
import com.japanesehelper.data.remote.dto.AgentInvariantsResponseDto
import com.japanesehelper.data.remote.dto.AgentLongTermMemoryDto
import com.japanesehelper.data.remote.dto.AgentMemoryResponseDto
import com.japanesehelper.data.remote.dto.AgentShortTermMemoryDto
import com.japanesehelper.data.remote.dto.AgentTaskErrorDto
import com.japanesehelper.data.remote.dto.AgentTaskRefusalDto
import com.japanesehelper.data.remote.dto.AgentTaskStateDto
import com.japanesehelper.data.remote.dto.AgentUserProfileDto
import com.japanesehelper.data.remote.dto.AgentUserProfileRequestDto
import com.japanesehelper.data.remote.dto.AgentWorkingMemoryDto
import com.japanesehelper.data.remote.dto.AgentUsageDto
import com.japanesehelper.domain.model.AgentContext
import com.japanesehelper.domain.model.AgentDigest
import com.japanesehelper.domain.model.AgentInvariant
import com.japanesehelper.domain.model.AgentInvariantCategory
import com.japanesehelper.domain.model.AgentLongTermMemory
import com.japanesehelper.domain.model.AgentMemory
import com.japanesehelper.domain.model.AgentMessage
import com.japanesehelper.domain.model.AgentMessageRole
import com.japanesehelper.domain.model.AgentPipeline
import com.japanesehelper.domain.model.AgentPipelineStage
import com.japanesehelper.domain.model.AgentPipelineStep
import com.japanesehelper.domain.model.AgentPipelineWord
import com.japanesehelper.domain.model.AgentReply
import com.japanesehelper.domain.model.AgentShortTermMemory
import com.japanesehelper.domain.model.AgentTaskRefusal
import com.japanesehelper.domain.model.AgentTaskState
import com.japanesehelper.domain.model.AgentTokenUsage
import com.japanesehelper.domain.model.AgentToolCall
import com.japanesehelper.domain.model.AgentUserProfile
import com.japanesehelper.domain.model.AgentWorkingMemory

fun AgentChatResponseDto.toDomain(): AgentReply {
    return AgentReply(
        text = response,
        usage = usage.toDomain(),
        toolCalls = toolCalls.orEmpty().mapNotNull { it.toDomain() },
        pipeline = toolCalls.orEmpty().toPipeline()
    )
}

/**
 * The chain, read out of the tool calls the backend already reports: which
 * stages it says it ran, and the few values from their results the readout
 * shows. Null when this answer was not a pipeline - an ordinary lookup, or
 * no tool at all.
 *
 * Nothing is decided here. The order is the order the backend sent, a stage
 * that is missing is missing because the backend stopped there, and no value
 * is computed from another: the words come from search's result, the summary
 * from summarize's, the file name from save_to_file's.
 */
fun List<AgentToolCallDto>.toPipeline(): AgentPipeline? {
    val stages = mapNotNull { call ->
        val stage = call.tool?.let { AgentPipelineStage.of(it) } ?: return@mapNotNull null
        stage to call
    }

    if (stages.isEmpty()) {
        return null
    }

    val results = stages.associate { (stage, call) -> stage to call.result.asObject() }
    val search = results[AgentPipelineStage.SEARCH]
    val summarize = results[AgentPipelineStage.SUMMARIZE]
    val saved = results[AgentPipelineStage.SAVE]

    return AgentPipeline(
        query = search.text("query").ifBlank { summarize.text("query") },
        steps = stages.map { (stage, call) ->
            AgentPipelineStep(
                stage = stage,
                tool = call.tool.orEmpty(),
                server = call.server.orEmpty(),
                ok = call.ok ?: true,
                error = call.error.orEmpty()
            )
        },
        found = search.array("matches").mapNotNull { it.asObject()?.toPipelineWord() },
        summary = summarize.text("summary"),
        fileName = saved.text("file_name"),
        filePath = saved.text("path")
    )
}

private fun JsonObject.toPipelineWord(): AgentPipelineWord? {
    val word = text("word")

    if (word.isBlank()) {
        return null
    }

    return AgentPipelineWord(
        word = word,
        reading = text("reading"),
        romaji = text("romaji"),
        meaning = text("meaning"),
        level = text("jlpt_level")
    )
}

/** A tool's result is only useful here when it is an object; a tool that
 * answered with a string or with nothing reads as absent. */
private fun JsonElement?.asObject(): JsonObject? = this?.takeIf { it.isJsonObject }?.asJsonObject

private fun JsonObject?.text(field: String): String =
    this?.get(field)?.takeIf { it.isJsonPrimitive }?.asString.orEmpty()

private fun JsonObject?.array(field: String): List<JsonElement> =
    (this?.get(field)?.takeIf { it.isJsonArray }?.asJsonArray ?: JsonArray()).toList()

/** An entry without a tool name says nothing that can be shown, so it is
 * dropped. Arguments are shown as text: a word, a kanji. */
fun AgentToolCallDto.toDomain(): AgentToolCall? {
    val name = tool?.takeIf { it.isNotBlank() } ?: return null

    return AgentToolCall(
        tool = name,
        arguments = arguments.orEmpty().mapValues { (_, value) -> value?.toString().orEmpty() },
        ok = ok ?: true,
        error = error.orEmpty()
    )
}

fun AgentUsageDto.toDomain(): AgentTokenUsage {
    return AgentTokenUsage(
        currentRequestTokens = currentRequestTokens,
        historyTokens = historyTokens,
        responseTokens = responseTokens,
        totalTokens = totalTokens
    )
}

fun AgentHistoryMessageDto.toDomain(): AgentMessage {
    val mappedRole = if (role == "user") AgentMessageRole.USER else AgentMessageRole.ASSISTANT
    return AgentMessage(role = mappedRole, content = content)
}

fun AgentHistoryResponseDto.toDomain(): List<AgentMessage> = messages.map { it.toDomain() }

fun AgentContextResponseDto.toDomain(): AgentContext {
    return AgentContext(
        strategy = strategy,
        branch = branch,
        branches = branches,
        checkpoints = checkpoints,
        facts = facts,
        messages = messages.map { it.toDomain() }
    )
}

fun AgentShortTermMemoryDto.toDomain(): AgentShortTermMemory =
    AgentShortTermMemory(messages = messages.map { it.toDomain() })

fun AgentWorkingMemoryDto.toDomain(): AgentWorkingMemory = AgentWorkingMemory(
    goals = goals,
    requirements = requirements,
    constraints = constraints,
    decisions = decisions
)

fun AgentLongTermMemoryDto.toDomain(): AgentLongTermMemory = AgentLongTermMemory(
    profile = profile,
    preferences = preferences,
    decisions = decisions,
    knowledge = knowledge
)

fun AgentMemoryResponseDto.toDomain(): AgentMemory = AgentMemory(
    shortTerm = shortTerm.toDomain(),
    working = working.toDomain(),
    longTerm = longTerm.toDomain()
)

fun AgentUserProfileDto.toDomain(): AgentUserProfile = AgentUserProfile(
    japaneseLevel = japaneseLevel,
    explanationStyle = explanationStyle,
    answerFormat = answerFormat,
    translationLanguage = translationLanguage
)

fun AgentUserProfile.toRequestDto(): AgentUserProfileRequestDto = AgentUserProfileRequestDto(
    japaneseLevel = japaneseLevel,
    explanationStyle = explanationStyle,
    answerFormat = answerFormat,
    translationLanguage = translationLanguage
)

fun AgentDigestDto.toDomain(): AgentDigest = AgentDigest(
    found = found == true,
    active = active == true,
    query = query.orEmpty(),
    intervalSeconds = intervalSeconds ?: 0,
    runs = runs ?: 0,
    failedRuns = failedRuns ?: 0,
    lastRun = lastRun.orEmpty(),
    itemsCollected = itemsCollected ?: 0,
    summary = summary.orEmpty()
)

fun AgentTaskStateDto.toDomain(): AgentTaskState = AgentTaskState(
    stage = taskStage,
    currentStep = currentStep,
    expectedAction = expectedAction,
    allowedNext = allowedNext,
    plan = plan.orEmpty(),
    validationPassed = validationPassed == true,
    nextRequirement = nextRequirement.orEmpty(),
    blocked = blocked?.toDomain()
)

fun AgentTaskRefusalDto.toDomain(): AgentTaskRefusal = AgentTaskRefusal(
    message = message.orEmpty(),
    currentStage = currentStage.orEmpty(),
    requestedStage = requestedStage.orEmpty(),
    requiredNext = requiredNext.orEmpty(),
    unmetCondition = unmetCondition.orEmpty()
)

/**
 * The refusal out of an error response body, or null if the body is not one.
 *
 * A refusal is the backend's answer, not an outage, so it is read rather
 * than shown as raw JSON - but only if it really is one: anything else
 * (a gateway error page, an empty body) stays an ordinary failure.
 */
private val refusalGson = Gson()

fun parseTaskRefusal(body: String): AgentTaskRefusal? = runCatching {
    refusalGson.fromJson(body, AgentTaskErrorDto::class.java)?.detail?.toDomain()
}.getOrNull()?.takeIf { it.currentStage.isNotBlank() }

/** A rule whose category this build does not know about is dropped rather
 * than shown under a made-up heading - the backend owns that list. */
fun AgentInvariantDto.toDomain(): AgentInvariant? =
    AgentInvariantCategory.fromWireName(category)?.let {
        AgentInvariant(id = id, category = it, rule = rule)
    }

fun AgentInvariantsResponseDto.toDomain(): List<AgentInvariant> = invariants.mapNotNull { it.toDomain() }
