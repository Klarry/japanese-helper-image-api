package com.japanesehelper.presentation.screens.aiAgentScreen.components

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.japanesehelper.R
import com.japanesehelper.domain.model.AgentPipeline
import com.japanesehelper.domain.model.AgentPipelineStage
import com.japanesehelper.domain.model.AgentPipelineStep
import com.japanesehelper.domain.model.AgentPipelineWord
import com.japanesehelper.presentation.theme.LocalAppPadding

/** How far the tool line is indented under the server it ran on. */
private val STEP_INDENT = 12.dp

/**
 * What the backend's orchestrator did, under the answer it produced: server
 * by server, which tool ran there and whether it completed, then whether the
 * whole run finished - and finally what it left behind: the words the search
 * found, the summary, and the file the result was saved to.
 *
 * It displays and nothing else. The device runs no chain, holds no MCP
 * client and knows neither the order of the stages nor which server offers
 * which tool: both names on every line are the backend's own words. A stage
 * is shown because the backend reported it, and the stages after a failed
 * one are missing here because the backend never ran them.
 */
@Composable
fun PipelineStatus(pipeline: AgentPipeline, modifier: Modifier = Modifier) {
    val reported = pipeline.steps.associateBy { it.stage }

    Surface(
        modifier = modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.medium,
        color = MaterialTheme.colorScheme.surfaceVariant,
        contentColor = MaterialTheme.colorScheme.onSurfaceVariant
    ) {
        Column(modifier = Modifier.padding(LocalAppPadding.current.half)) {
            ReadoutCaption(stringResource(R.string.ai_agent_pipeline_title))

            AgentPipelineStage.values().forEach { stage ->
                Stage(stage = stage, step = reported[stage])
            }

            Text(
                text = stringResource(
                    if (pipeline.completed) {
                        R.string.ai_agent_pipeline_done
                    } else {
                        R.string.ai_agent_pipeline_unfinished
                    }
                ),
                modifier = Modifier.padding(top = LocalAppPadding.current.quarter),
                style = MaterialTheme.typography.labelMedium,
                color = if (pipeline.completed) {
                    MaterialTheme.colorScheme.primary
                } else {
                    MaterialTheme.colorScheme.error
                }
            )

            pipeline.failed?.let { step ->
                Outcome(
                    stringResource(
                        R.string.ai_agent_pipeline_stopped,
                        step.name(),
                        step.error.ifBlank { stringResource(R.string.ai_agent_pipeline_no_reason) }
                    )
                )
            }

            pipeline.found.takeIf { it.isNotEmpty() }?.let { words ->
                Outcome(stringResource(R.string.ai_agent_pipeline_found, words.joinToString("; ") { it.readable() }))
            }

            pipeline.summary.takeIf { it.isNotBlank() }?.let { summary ->
                Outcome(stringResource(R.string.ai_agent_pipeline_summary, summary), maxLines = 4)
            }

            pipeline.fileName.takeIf { it.isNotBlank() }?.let { name ->
                Outcome(stringResource(R.string.ai_agent_pipeline_saved, name))
            }
        }
    }
}

/** One server and, indented under it, what its tool did there. A stage the
 * backend never got to has neither. */
@Composable
private fun Stage(stage: AgentPipelineStage, step: AgentPipelineStep?) {
    Text(
        text = stringResource(
            when {
                step == null -> R.string.ai_agent_pipeline_server_skipped
                step.ok -> R.string.ai_agent_pipeline_server_ok
                else -> R.string.ai_agent_pipeline_server_failed
            },
            step?.serverLabel() ?: stringResource(stage.label())
        ),
        style = MaterialTheme.typography.labelMedium,
        color = when {
            step == null -> MaterialTheme.colorScheme.outline
            step.ok -> MaterialTheme.colorScheme.primary
            else -> MaterialTheme.colorScheme.error
        }
    )

    if (step != null) {
        Text(
            text = stringResource(
                if (step.ok) R.string.ai_agent_pipeline_step_ok else R.string.ai_agent_pipeline_step_failed,
                step.name()
            ),
            modifier = Modifier.padding(start = STEP_INDENT),
            style = MaterialTheme.typography.bodySmall,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}

@Composable
private fun Outcome(text: String, maxLines: Int = 2) {
    Text(
        text = text,
        modifier = Modifier.padding(top = LocalAppPadding.current.quarter),
        style = MaterialTheme.typography.bodySmall,
        maxLines = maxLines,
        overflow = TextOverflow.Ellipsis
    )
}

/** The tool the backend named, or the stage's own name when an older backend
 * sent none. */
@Composable
private fun AgentPipelineStep.name(): String = tool.ifBlank { stringResource(stage.label()) }

/** "japanese-data" -> "Japanese Data Server". A spelling of the name the
 * backend sent, not a list of servers: anything it reports reads the same
 * way, and an older backend that sends none falls back to the stage. */
@Composable
private fun AgentPipelineStep.serverLabel(): String {
    if (server.isBlank()) {
        return stringResource(stage.label())
    }

    val words = server.split('-', '_', ' ').filter { it.isNotBlank() }.joinToString(" ") { word ->
        word.replaceFirstChar { it.uppercase() }
    }

    return stringResource(R.string.ai_agent_pipeline_server_name, words)
}

private fun AgentPipelineStage.label(): Int = when (this) {
    AgentPipelineStage.SEARCH -> R.string.ai_agent_pipeline_search
    AgentPipelineStage.SUMMARIZE -> R.string.ai_agent_pipeline_summarize
    AgentPipelineStage.SAVE -> R.string.ai_agent_pipeline_save
}

/** "学習 · がくしゅう (gakushū) · study, learning · N3", with whatever the
 * dictionary left empty left out. */
private fun AgentPipelineWord.readable(): String {
    val readAs = when {
        reading.isNotBlank() && romaji.isNotBlank() -> "$reading ($romaji)"
        else -> reading.ifBlank { romaji }
    }

    return listOf(word, readAs, meaning, level).filter { it.isNotBlank() }.joinToString(" · ")
}
