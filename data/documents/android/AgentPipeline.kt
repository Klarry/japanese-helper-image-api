package com.japanesehelper.domain.model

/**
 * The three MCP tools the backend runs as one chain, as the screen reads
 * them back: search, then summarize on what search found, then save_to_file
 * on that. Since Day 20 each of them runs on a server of its own, and the
 * backend says which - so the readout can show the route as well as the
 * result.
 *
 * Reported, not run. The device holds no MCP client, calls no tool, knows
 * nothing about the order the stages belong in and nothing about which
 * server offers what - the backend decides all of it, runs the chain, and
 * says in its answer what happened. This is that report, in the shape the
 * screen shows it.
 */
enum class AgentPipelineStage(val tool: String) {
    SEARCH("search"),
    SUMMARIZE("summarize"),
    SAVE("save_to_file");

    companion object {
        /** The stage a reported tool name belongs to, or null when the call
         * was an ordinary lookup rather than part of the chain. */
        fun of(tool: String): AgentPipelineStage? = values().firstOrNull { it.tool == tool }
    }
}

/** One stage the backend reported: which tool ran, on which of its servers,
 * and how it went. Both names come from the backend - the device has no idea
 * which server offers what, and is not supposed to. */
data class AgentPipelineStep(
    val stage: AgentPipelineStage,
    val tool: String = "",
    val server: String = "",
    val ok: Boolean = true,
    val error: String = ""
)

/** One dictionary entry the first stage found. */
data class AgentPipelineWord(
    val word: String,
    val reading: String = "",
    val romaji: String = "",
    val meaning: String = "",
    val level: String = ""
)

data class AgentPipeline(
    val query: String = "",
    val steps: List<AgentPipelineStep> = emptyList(),
    /** What search found. */
    val found: List<AgentPipelineWord> = emptyList(),
    /** What summarize made of it. */
    val summary: String = "",
    /** Where save_to_file put it. */
    val fileName: String = "",
    val filePath: String = ""
) {
    /** Every stage ran and none of them failed. A chain that stopped early
     * has fewer steps, because the backend does not attempt what comes after
     * a failure. */
    val completed: Boolean
        get() = steps.size == AgentPipelineStage.values().size && steps.all { it.ok }

    /** The stage that stopped the chain, if one did. */
    val failed: AgentPipelineStep?
        get() = steps.firstOrNull { !it.ok }
}
