package com.japanesehelper.presentation.screens.aiAgentScreen

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.navigation.NavController
import com.japanesehelper.R
import com.japanesehelper.domain.model.AgentContextStrategy
import com.japanesehelper.presentation.screens.aiAgentScreen.components.AgentConversation
import com.japanesehelper.presentation.screens.aiAgentScreen.components.AskAgentButton
import com.japanesehelper.presentation.screens.aiAgentScreen.components.BranchingControls
import com.japanesehelper.presentation.screens.aiAgentScreen.components.ClearHistoryButton
import com.japanesehelper.presentation.screens.aiAgentScreen.components.ContextSection
import com.japanesehelper.presentation.screens.aiAgentScreen.components.ContextStrategyTabRow
import com.japanesehelper.presentation.screens.aiAgentScreen.components.CreateBranchDialog
import com.japanesehelper.presentation.screens.aiAgentScreen.components.ApprovePlanDialog
import com.japanesehelper.presentation.screens.aiAgentScreen.components.EditInvariantsDialog
import com.japanesehelper.presentation.screens.aiAgentScreen.components.EditProfileDialog
import com.japanesehelper.presentation.screens.aiAgentScreen.components.InvariantsSection
import com.japanesehelper.presentation.screens.aiAgentScreen.components.MemoryLayersSection
import com.japanesehelper.presentation.screens.aiAgentScreen.components.PeriodicTaskSection
import com.japanesehelper.presentation.screens.aiAgentScreen.components.TaskStateSection
import com.japanesehelper.presentation.screens.aiAgentScreen.components.TokenUsageSection
import com.japanesehelper.presentation.screens.aiAgentScreen.components.UserProfileSection
import com.japanesehelper.presentation.screens.homeScreen.components.ErrorWithRetry
import com.japanesehelper.presentation.screens.homeScreen.components.ScreenTopBar
import com.japanesehelper.presentation.theme.LocalAppPadding
import com.japanesehelper.presentation.viewmodel.AiAgentViewModel
import com.japanesehelper.presentation.viewmodel.screendata.AgentHistoryUiState

/** The input grows with a long question, but never so far that it crowds out
 * the conversation above it. */
private const val MAX_INPUT_LINES = 4

/** The slice of the screen the readouts get. Fixed rather than however tall
 * they happen to be, so the conversation above never resizes. */
private val READOUTS_HEIGHT = 156.dp

/**
 * A persistent chat with the backend's JapaneseLearningAgent, laid out as one
 * chat window: the conversation owns the height of the screen and scrolls on
 * its own, while the input and the readouts stay put underneath it.
 *
 * The Context Strategy tabs choose how the backend assembles what it sends -
 * the newest messages only, a key-value memory of what matters, the branch
 * being talked on, or the three memory layers kept apart. Choosing is all
 * this screen does: it never trims the history, builds facts or decides what
 * belongs in a memory layer, and what it shows under the input is what the
 * backend reports it would send. The existing Token Usage block is what makes
 * the difference between the strategies visible.
 *
 * The User Profile block underneath is a setting rather than a strategy: the
 * backend applies it to every request whatever the strategy, so the learner
 * never repeats their level or format in a message. Switching between the two
 * profiles and asking the same question again is the whole comparison.
 *
 * The Task State block below it shows where the work has got to - planning,
 * execution, validation, done. The stages and the moves between them are the
 * backend's state machine; the screen reads it and can end the task, and holds
 * no idea of its own about which stage follows which.
 *
 * The Invariants block under that lists the rules the agent may not break, and
 * the dialog behind Edit changes them. Whether a request breaks one is decided
 * on the backend and answered in the chat like any other reply - nothing here
 * inspects a message.
 *
 * All of those blocks live in one pane of a fixed height that scrolls inside
 * itself, so the conversation is exactly as tall on the last turn as on the
 * first: a strategy with more to report, or another invariant, no longer takes
 * the screen away from the chat.
 */
@Composable
fun AiAgentScreen(
    navController: NavController,
    viewModel: AiAgentViewModel = hiltViewModel()
) {
    val state by viewModel.state.collectAsState()

    // The periodic task keeps collecting on the backend. While this screen is
    // on show, its block re-reads itself, so a run that has just happened is
    // there without leaving the screen and coming back.
    DisposableEffect(Unit) {
        viewModel.startWatchingPeriodicTask()
        onDispose { viewModel.stopWatchingPeriodicTask() }
    }

    val padding = LocalAppPadding.current
    val loadedMessages = (state.history as? AgentHistoryUiState.Loaded)?.messages

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        topBar = {
            ScreenTopBar(
                title = stringResource(R.string.ai_agent_title),
                onBack = { navController.popBackStack() }
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .padding(innerPadding)
                .fillMaxSize()
                .imePadding()
        ) {
            ContextStrategyTabRow(
                selected = state.strategy,
                onSelected = viewModel::onStrategySelected
            )

            AgentConversation(
                historyState = state.history,
                onRetryLoad = viewModel::loadHistory,
                modifier = Modifier.weight(1f)
            )

            HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = padding.default, vertical = padding.half),
                verticalArrangement = Arrangement.spacedBy(padding.half)
            ) {
                OutlinedTextField(
                    value = state.message,
                    onValueChange = viewModel::onMessageChanged,
                    modifier = Modifier.fillMaxWidth(),
                    placeholder = { Text(stringResource(R.string.ai_agent_input_placeholder)) },
                    textStyle = MaterialTheme.typography.bodyLarge,
                    maxLines = MAX_INPUT_LINES
                )

                AskAgentButton(
                    canAsk = state.message.isNotBlank(),
                    isLoading = state.isSending,
                    onAsk = viewModel::send
                )
            }

            // The readouts get a fixed slice of the screen and scroll inside
            // it. They are why the conversation used to shrink: a strategy
            // with more to report, or a few more invariants, took the room
            // from the chat. Now nothing below the conversation changes size,
            // so the chat stays exactly as tall as it was when the screen
            // opened.
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(READOUTS_HEIGHT)
                    .padding(horizontal = padding.default, vertical = padding.half),
                horizontalArrangement = Arrangement.spacedBy(padding.default),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .fillMaxHeight()
                        .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(padding.quarter)
                ) {
                    val sendError = state.sendError
                    if (sendError != null) {
                        ErrorWithRetry(message = sendError, onRetry = viewModel::send)
                    }

                    if (state.strategy == AgentContextStrategy.BRANCHING) {
                        BranchingControls(
                            context = state.context,
                            isWorking = state.isBranchWorking,
                            onSwitchBranch = viewModel::switchBranch,
                            onCreateCheckpoint = viewModel::createCheckpoint,
                            onCreateBranch = viewModel::openNewBranch
                        )
                    }

                    UserProfileSection(
                        profile = state.profile,
                        isWorking = state.isProfileWorking,
                        onPresetSelected = viewModel::applyPreset,
                        onEdit = viewModel::openProfileEditor
                    )

                    val profileError = state.profileError
                    if (profileError != null) {
                        ReadoutError(profileError)
                    }

                    PeriodicTaskSection(digest = state.digest)

                    TaskStateSection(
                        taskState = state.taskState,
                        refusal = state.taskRefusal,
                        isWorking = state.isTaskWorking,
                        onRequestStage = viewModel::requestTaskTransition,
                        onApprovePlan = viewModel::startEditingPlan,
                        onValidationPassed = { viewModel.recordTaskValidation(passed = true) },
                        onClear = viewModel::clearTaskState
                    )

                    InvariantsSection(
                        invariants = state.invariants,
                        isWorking = state.isInvariantsWorking,
                        onEdit = viewModel::openInvariantEditor
                    )

                    val invariantsError = state.invariantsError
                    if (invariantsError != null) {
                        ReadoutError(invariantsError)
                    }

                    val lastUsage = state.lastUsage
                    if (lastUsage != null) {
                        TokenUsageSection(usage = lastUsage)
                    }

                    ContextSection(strategy = state.strategy, context = state.context)

                    if (state.strategy == AgentContextStrategy.LAYERED_MEMORY) {
                        MemoryLayersSection(
                            memory = state.memory,
                            isWorking = state.isMemoryWorking,
                            onClearLayer = viewModel::clearMemoryLayer
                        )
                    }

                    val contextError = state.contextError
                    if (contextError != null) {
                        ReadoutError(contextError)
                    }

                    val clearError = state.clearHistoryError
                    if (clearError != null) {
                        ErrorWithRetry(message = clearError, onRetry = viewModel::clearHistory)
                    }
                }

                ClearHistoryButton(
                    enabled = !loadedMessages.isNullOrEmpty(),
                    isLoading = state.isClearingHistory,
                    onClear = viewModel::clearHistory
                )
            }
        }

        val invariantEditor = state.invariantEditor
        if (invariantEditor != null) {
            EditInvariantsDialog(
                invariants = state.invariants.orEmpty(),
                editor = invariantEditor,
                isWorking = state.isInvariantsWorking,
                onSelect = viewModel::onInvariantSelected,
                onDelete = viewModel::deleteInvariant,
                onRuleChanged = viewModel::onInvariantRuleChanged,
                onCategorySelected = viewModel::onInvariantCategorySelected,
                onSave = viewModel::saveInvariant,
                onDismiss = viewModel::dismissInvariantEditor
            )
        }

        val profileEditor = state.profileEditor
        if (profileEditor != null) {
            EditProfileDialog(
                profile = profileEditor.profile,
                onProfileChanged = viewModel::onProfileEdited,
                onConfirm = viewModel::confirmProfileEdit,
                onDismiss = viewModel::dismissProfileEditor
            )
        }

        val planEditor = state.planEditor
        if (planEditor != null) {
            ApprovePlanDialog(
                plan = planEditor,
                isWorking = state.isTaskWorking,
                onPlanChanged = viewModel::onPlanChanged,
                onApprove = viewModel::approveTaskPlan,
                onDismiss = viewModel::stopEditingPlan
            )
        }

        val newBranch = state.newBranch
        if (newBranch != null) {
            CreateBranchDialog(
                state = newBranch,
                checkpoints = state.context?.checkpoints.orEmpty(),
                onNameChanged = viewModel::onNewBranchNameChanged,
                onCheckpointSelected = viewModel::onNewBranchCheckpointSelected,
                onConfirm = viewModel::confirmNewBranch,
                onDismiss = viewModel::dismissNewBranch
            )
        }
    }
}

/** One line of red under a readout - the same shape for every block. */
@Composable
private fun ReadoutError(message: String) {
    Text(
        text = message,
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.error
    )
}
