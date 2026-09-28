package com.japanesehelper.presentation.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.japanesehelper.domain.model.AgentContextStrategy
import com.japanesehelper.domain.model.AgentInvariantCategory
import com.japanesehelper.domain.model.AgentMemoryLayer
import com.japanesehelper.domain.model.AgentMessage
import com.japanesehelper.domain.model.AgentMessageRole
import com.japanesehelper.domain.model.AgentProfilePreset
import com.japanesehelper.domain.model.AgentTaskStage
import com.japanesehelper.domain.model.AgentTaskState
import com.japanesehelper.domain.model.AgentTaskTransitionRefused
import com.japanesehelper.domain.model.AgentUserProfile
import com.japanesehelper.domain.repository.AgentRepository
import com.japanesehelper.presentation.viewmodel.screendata.AgentHistoryUiState
import com.japanesehelper.presentation.viewmodel.screendata.AiAgentScreenState
import com.japanesehelper.presentation.viewmodel.screendata.InvariantEditorState
import com.japanesehelper.presentation.viewmodel.screendata.NewBranchState
import com.japanesehelper.presentation.viewmodel.screendata.ProfileEditorState
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import retrofit2.HttpException
import javax.inject.Inject

/** How often the periodic-task block re-reads itself while the screen is on
 * show. Short enough that a run which happened on the backend appears while
 * the learner is still looking at the block, long enough that a readout does
 * not turn into a stream of requests. */
private const val DIGEST_POLL_MILLIS = 5_000L

/**
 * The AI Agent screen is a simple chat: history is loaded once on open, every
 * sent message is appended (with the agent's reply) to the same list, and
 * "Clear History" empties it.
 *
 * Everything about context management belongs to the backend. This class
 * names a strategy, asks for checkpoints and branches, and displays what the
 * backend reports it would send - it never trims the history or builds facts
 * itself. After anything that could change that, it reloads the context
 * rather than guessing at the new state.
 */
@HiltViewModel
class AiAgentViewModel @Inject constructor(
    private val agentRepository: AgentRepository
) : ViewModel() {

    private val _state = MutableStateFlow(AiAgentScreenState())
    val state: StateFlow<AiAgentScreenState> = _state

    /** The repeat that keeps the periodic-task block current while the screen
     * is open. It only re-reads what the backend reports - no part of the
     * schedule itself lives on the device. */
    private var digestWatcher: Job? = null

    // Opening the screen used to fire five requests at once. On a slow or
    // busy backend they competed with each other and whichever lost came
    // back empty, which is what made the screen load "every other time".
    // One coroutine, in order of what the reader needs first: the
    // conversation, then the readouts.
    init {
        viewModelScope.launch {
            loadHistoryNow()
            alignStrategyAndLoadContext()
            refreshProfile()
            refreshTaskState()
            refreshInvariants()
            refreshDigest()
        }
    }

    fun loadHistory() {
        viewModelScope.launch { loadHistoryNow() }
    }

    private suspend fun loadHistoryNow() {
        _state.value = _state.value.copy(history = AgentHistoryUiState.Loading)

        try {
            val messages = agentRepository.getHistory()
            _state.value = _state.value.copy(history = AgentHistoryUiState.Loaded(messages))
        } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
            _state.value = _state.value.copy(history = AgentHistoryUiState.Error(e.toErrorMessage()))
        }
    }

    fun loadContext() = loadContext(alignStrategy = false)

    fun onMessageChanged(message: String) {
        _state.value = _state.value.copy(message = message)
    }

    fun onStrategySelected(strategy: AgentContextStrategy) {
        if (strategy == _state.value.strategy) return

        _state.value = _state.value.copy(strategy = strategy)

        viewModelScope.launch {
            try {
                agentRepository.setStrategy(strategy)
            } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
                _state.value = _state.value.copy(contextError = e.toErrorMessage())
                return@launch
            }

            refreshContext()
        }
    }

    fun send() {
        val message = _state.value.message.trim()

        if (message.isEmpty()) return
        if (_state.value.isSending) return

        // Read once, so a strategy switched while this request is in flight
        // applies to the next one instead of mislabelling this one.
        val strategy = _state.value.strategy

        _state.value = _state.value.copy(isSending = true, sendError = null)

        viewModelScope.launch {
            try {
                val reply = agentRepository.chat(message, strategy)
                val updatedMessages = currentMessages() +
                    AgentMessage(role = AgentMessageRole.USER, content = message) +
                    AgentMessage(
                        role = AgentMessageRole.ASSISTANT,
                        content = reply.text,
                        toolCalls = reply.toolCalls,
                        pipeline = reply.pipeline
                    )

                _state.value = _state.value.copy(
                    message = "",
                    isSending = false,
                    history = AgentHistoryUiState.Loaded(updatedMessages),
                    lastUsage = reply.usage
                )
            } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
                _state.value = _state.value.copy(isSending = false, sendError = e.toErrorMessage())
                return@launch
            }

            refreshContext()
        }
    }

    fun clearHistory() {
        if (_state.value.isClearingHistory) return

        _state.value = _state.value.copy(isClearingHistory = true, clearHistoryError = null)

        viewModelScope.launch {
            try {
                agentRepository.clearHistory()
                _state.value = _state.value.copy(
                    isClearingHistory = false,
                    history = AgentHistoryUiState.Loaded(emptyList()),
                    lastUsage = null
                )
            } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
                _state.value = _state.value.copy(
                    isClearingHistory = false,
                    clearHistoryError = e.toErrorMessage()
                )
                return@launch
            }

            refreshContext()
        }
    }

    // --- branching ---------------------------------------------------------

    fun createCheckpoint() {
        if (_state.value.isBranchWorking) return

        _state.value = _state.value.copy(isBranchWorking = true, contextError = null)

        viewModelScope.launch {
            try {
                agentRepository.createCheckpoint()
            } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
                _state.value = _state.value.copy(isBranchWorking = false, contextError = e.toErrorMessage())
                return@launch
            }

            refreshContext()
            _state.value = _state.value.copy(isBranchWorking = false)
        }
    }

    fun openNewBranch() {
        // The newest checkpoint is the one a fork usually means; any other can
        // still be picked in the dialog.
        val checkpoint = _state.value.context?.checkpoints?.lastOrNull().orEmpty()
        _state.value = _state.value.copy(newBranch = NewBranchState(checkpoint = checkpoint))
    }

    fun onNewBranchNameChanged(name: String) {
        val dialog = _state.value.newBranch ?: return
        _state.value = _state.value.copy(newBranch = dialog.copy(name = name))
    }

    fun onNewBranchCheckpointSelected(checkpoint: String) {
        val dialog = _state.value.newBranch ?: return
        _state.value = _state.value.copy(newBranch = dialog.copy(checkpoint = checkpoint))
    }

    fun dismissNewBranch() {
        _state.value = _state.value.copy(newBranch = null)
    }

    fun confirmNewBranch() {
        val dialog = _state.value.newBranch ?: return
        val name = dialog.name.trim()

        if (name.isEmpty() || dialog.checkpoint.isEmpty()) return

        _state.value = _state.value.copy(newBranch = null, isBranchWorking = true, contextError = null)

        viewModelScope.launch {
            try {
                agentRepository.createBranch(name, dialog.checkpoint)
            } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
                _state.value = _state.value.copy(isBranchWorking = false, contextError = e.toErrorMessage())
                return@launch
            }

            refreshContext()
            _state.value = _state.value.copy(isBranchWorking = false)
        }
    }

    fun switchBranch(name: String) {
        if (_state.value.isBranchWorking || name == _state.value.context?.branch) return

        _state.value = _state.value.copy(isBranchWorking = true, contextError = null)

        viewModelScope.launch {
            try {
                agentRepository.switchBranch(name)
            } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
                _state.value = _state.value.copy(isBranchWorking = false, contextError = e.toErrorMessage())
                return@launch
            }

            // The branch that was switched to has its own conversation, so the
            // chat itself has to be reloaded, not just the context readout.
            loadHistory()
            refreshContext()
            _state.value = _state.value.copy(isBranchWorking = false, lastUsage = null)
        }
    }

    // --- invariants --------------------------------------------------------

    fun loadInvariants() {
        viewModelScope.launch { refreshInvariants() }
    }

    fun openInvariantEditor() {
        _state.value = _state.value.copy(invariantEditor = InvariantEditorState())
    }

    /** Pick an existing rule to change. Its text and category go into the
     * editor, and saving sends it back under the same id. */
    fun onInvariantSelected(id: String) {
        val existing = _state.value.invariants?.firstOrNull { it.id == id } ?: return
        _state.value = _state.value.copy(
            invariantEditor = InvariantEditorState(
                editingId = existing.id,
                category = existing.category,
                rule = existing.rule
            )
        )
    }

    fun onInvariantRuleChanged(rule: String) {
        val editor = _state.value.invariantEditor ?: return
        _state.value = _state.value.copy(invariantEditor = editor.copy(rule = rule))
    }

    fun onInvariantCategorySelected(category: AgentInvariantCategory) {
        val editor = _state.value.invariantEditor ?: return
        _state.value = _state.value.copy(invariantEditor = editor.copy(category = category))
    }

    fun dismissInvariantEditor() {
        _state.value = _state.value.copy(invariantEditor = null)
    }

    fun saveInvariant() {
        val editor = _state.value.invariantEditor ?: return
        val rule = editor.rule.trim()

        if (rule.isEmpty() || _state.value.isInvariantsWorking) return

        _state.value = _state.value.copy(isInvariantsWorking = true, invariantsError = null)

        viewModelScope.launch {
            try {
                val saved = agentRepository.saveInvariant(editor.editingId, editor.category, rule)
                _state.value = _state.value.copy(
                    isInvariantsWorking = false,
                    invariants = saved,
                    invariantEditor = null
                )
            } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
                _state.value = _state.value.copy(
                    isInvariantsWorking = false,
                    invariantsError = e.toErrorMessage()
                )
            }
        }
    }

    fun deleteInvariant(id: String) {
        if (_state.value.isInvariantsWorking) return

        _state.value = _state.value.copy(isInvariantsWorking = true, invariantsError = null)

        viewModelScope.launch {
            try {
                val remaining = agentRepository.deleteInvariant(id)
                _state.value = _state.value.copy(
                    isInvariantsWorking = false,
                    invariants = remaining,
                    invariantEditor = _state.value.invariantEditor?.takeIf { it.editingId != id }
                )
            } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
                _state.value = _state.value.copy(
                    isInvariantsWorking = false,
                    invariantsError = e.toErrorMessage()
                )
            }
        }
    }

    private suspend fun refreshInvariants() {
        try {
            _state.value = _state.value.copy(
                invariants = agentRepository.getInvariants(),
                invariantsError = null
            )
        } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
            _state.value = _state.value.copy(invariantsError = e.toErrorMessage())
        }
    }

    // --- task state --------------------------------------------------------

    fun loadTaskState() {
        viewModelScope.launch { refreshTaskState() }
    }

    /** End the task. The backend leaves the conversation, the memory layers
     * and the profile alone, so its answer is the whole new state. */
    fun clearTaskState() {
        if (_state.value.isTaskWorking) return

        _state.value = _state.value.copy(isTaskWorking = true, contextError = null)

        viewModelScope.launch {
            try {
                val cleared = agentRepository.clearTaskState()
                _state.value = _state.value.copy(
                    isTaskWorking = false,
                    taskState = cleared,
                    taskRefusal = cleared.blocked
                )
            } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
                _state.value = _state.value.copy(isTaskWorking = false, contextError = e.toErrorMessage())
            }
        }
    }

    /**
     * Ask the backend to move the task to [stage].
     *
     * Every stage is offered, including the ones the task cannot move to:
     * whether a move is allowed is the backend's answer, and asking is how
     * the device finds out. A refusal is shown as the backend worded it and
     * leaves the displayed state alone - because nothing moved.
     */
    fun requestTaskTransition(stage: AgentTaskStage) {
        runTaskRequest { agentRepository.requestTaskTransition(stage) }
    }

    /** Record the plan the task will be executed by, and close the editor. */
    fun approveTaskPlan() {
        val plan = _state.value.planEditor?.trim().orEmpty()

        if (plan.isEmpty()) return

        _state.value = _state.value.copy(planEditor = null)
        runTaskRequest { agentRepository.approveTaskPlan(plan) }
    }

    fun startEditingPlan() {
        _state.value = _state.value.copy(planEditor = _state.value.taskState?.plan.orEmpty())
    }

    fun onPlanChanged(plan: String) {
        if (_state.value.planEditor == null) return

        _state.value = _state.value.copy(planEditor = plan)
    }

    fun stopEditingPlan() {
        _state.value = _state.value.copy(planEditor = null)
    }

    /** Record how validation went. Sent from the device, judged on the
     * backend: recording it outside validation is refused there. */
    fun recordTaskValidation(passed: Boolean) {
        runTaskRequest { agentRepository.recordTaskValidation(passed) }
    }

    private fun runTaskRequest(request: suspend () -> AgentTaskState) {
        if (_state.value.isTaskWorking) return

        _state.value = _state.value.copy(isTaskWorking = true, taskRefusal = null, contextError = null)

        viewModelScope.launch {
            try {
                val updated = request()
                _state.value = _state.value.copy(
                    isTaskWorking = false,
                    taskState = updated,
                    taskRefusal = updated.blocked
                )
            } catch (e: AgentTaskTransitionRefused) {
                _state.value = _state.value.copy(isTaskWorking = false, taskRefusal = e.refusal)
            } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
                _state.value = _state.value.copy(isTaskWorking = false, contextError = e.toErrorMessage())
            }
        }
    }

    /** Keep the periodic-task block in step with the backend while the screen
     * is on show. The task runs there whether or not anyone is looking; before
     * this, the block only caught up when the screen was reopened or another
     * message was sent. */
    fun startWatchingPeriodicTask() {
        if (digestWatcher?.isActive == true) {
            return
        }
        digestWatcher = viewModelScope.launch {
            while (isActive) {
                delay(DIGEST_POLL_MILLIS)
                // A send is already followed by a refresh, and a readout is
                // not worth competing with the message for the backend.
                if (!_state.value.isSending) {
                    refreshDigest()
                }
            }
        }
    }

    /** Stop re-reading it once the screen is gone. */
    fun stopWatchingPeriodicTask() {
        digestWatcher?.cancel()
        digestWatcher = null
    }

    /** Read the periodic digest. Failing to read it is not worth an error on
     * screen: the block simply stays as it was, and the next refresh tries
     * again. */
    private suspend fun refreshDigest() {
        try {
            _state.value = _state.value.copy(digest = agentRepository.getDigest())
        } catch (@Suppress("TooGenericExceptionCaught", "SwallowedException") e: Exception) {
            // Left as it was on purpose: a readout that could not be refreshed
            // is not worth an error message over the conversation.
        }
    }

    private suspend fun refreshTaskState() {
        try {
            val taskState = agentRepository.getTaskState()
            // The refusal comes back with the state, so the reason a task did
            // not advance is still on the screen after the app is reopened.
            _state.value = _state.value.copy(
                taskState = taskState,
                taskRefusal = taskState.blocked,
                contextError = null
            )
        } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
            _state.value = _state.value.copy(contextError = e.toErrorMessage())
        }
    }

    // --- user profile ------------------------------------------------------

    fun loadProfile() {
        viewModelScope.launch { refreshProfile() }
    }

    /** Switch to one of the two profiles the comparison is made with. One
     * tap so the same question can be asked again immediately. */
    fun applyPreset(preset: AgentProfilePreset) {
        saveProfile(preset.profile)
    }

    fun openProfileEditor() {
        _state.value = _state.value.copy(
            profileEditor = ProfileEditorState(_state.value.profile ?: AgentUserProfile())
        )
    }

    fun onProfileEdited(profile: AgentUserProfile) {
        val editor = _state.value.profileEditor ?: return
        _state.value = _state.value.copy(profileEditor = editor.copy(profile = profile))
    }

    fun dismissProfileEditor() {
        _state.value = _state.value.copy(profileEditor = null)
    }

    fun confirmProfileEdit() {
        val editor = _state.value.profileEditor ?: return
        _state.value = _state.value.copy(profileEditor = null)
        saveProfile(editor.profile)
    }

    private fun saveProfile(profile: AgentUserProfile) {
        if (_state.value.isProfileWorking) return

        _state.value = _state.value.copy(isProfileWorking = true, profileError = null)

        viewModelScope.launch {
            try {
                val saved = agentRepository.updateProfile(profile)
                _state.value = _state.value.copy(isProfileWorking = false, profile = saved)
            } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
                _state.value = _state.value.copy(isProfileWorking = false, profileError = e.toErrorMessage())
            }
        }
    }

    private suspend fun refreshProfile() {
        try {
            _state.value = _state.value.copy(profile = agentRepository.getProfile(), profileError = null)
        } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
            _state.value = _state.value.copy(profileError = e.toErrorMessage())
        }
    }

    // --- memory layers -----------------------------------------------------

    /**
     * Empty one layer. The backend leaves the other two alone, so the answer
     * it returns is the whole new state of the memory - there is nothing to
     * reconcile here. Clearing short-term memory also empties the
     * conversation, so the chat is reloaded with it.
     */
    fun clearMemoryLayer(layer: AgentMemoryLayer) {
        if (_state.value.isMemoryWorking) return

        _state.value = _state.value.copy(isMemoryWorking = true, contextError = null)

        viewModelScope.launch {
            val memory = try {
                agentRepository.clearMemoryLayer(layer)
            } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
                _state.value = _state.value.copy(isMemoryWorking = false, contextError = e.toErrorMessage())
                return@launch
            }

            _state.value = _state.value.copy(isMemoryWorking = false, memory = memory)

            if (layer == AgentMemoryLayer.SHORT_TERM) {
                _state.value = _state.value.copy(
                    history = AgentHistoryUiState.Loaded(memory.shortTerm.messages),
                    lastUsage = null
                )
            }

            // The answer already carries the whole new memory, so only the
            // context readout is left to bring up to date.
            refreshContextOnly()
        }
    }

    // --- internals ---------------------------------------------------------

    private fun loadContext(alignStrategy: Boolean) {
        viewModelScope.launch {
            if (alignStrategy) alignStrategyAndLoadContext() else refreshContext()
        }
    }

    private suspend fun alignStrategyAndLoadContext() {
        val context = try {
            agentRepository.getContext()
        } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
            _state.value = _state.value.copy(contextError = e.toErrorMessage())
            return
        }

        val known = AgentContextStrategy.fromWireName(context.strategy)

        if (known != null) {
            _state.value = _state.value.copy(context = context, strategy = known, contextError = null)
            refreshMemory()
            return
        }

        // The backend is on a strategy this screen does not offer. Put it
        // on the one shown here, so what is displayed and what would be
        // sent cannot disagree.
        try {
            agentRepository.setStrategy(_state.value.strategy)
        } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
            _state.value = _state.value.copy(context = context, contextError = e.toErrorMessage())
            return
        }

        refreshContext()
    }

    private suspend fun refreshContext() {
        refreshContextOnly()
        refreshMemory()
        // Any message can move the task on, so the stage is re-read wherever
        // the rest of the readouts are.
        refreshTaskState()
        // The rules change only when someone edits them, so they are not
        // re-read on every turn - that would be a fourth request after each
        // message for something that did not move. They are re-read when
        // they are missing, which is what gets them back after a read that
        // failed while the backend was still starting.
        if (_state.value.invariants == null) {
            refreshInvariants()
        }
        // The periodic task runs on the backend whether or not anyone is
        // looking; re-reading it after a message is how the screen catches up
        // with the runs that happened in between.
        refreshDigest()
    }

    private suspend fun refreshContextOnly() {
        try {
            _state.value = _state.value.copy(context = agentRepository.getContext(), contextError = null)
        } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
            _state.value = _state.value.copy(contextError = e.toErrorMessage())
        }
    }

    /** Only the layered-memory strategy writes to the layers, so only it
     * needs them read back - the other strategies would be asking for a
     * readout they never show. */
    private suspend fun refreshMemory() {
        if (_state.value.strategy != AgentContextStrategy.LAYERED_MEMORY) {
            _state.value = _state.value.copy(memory = null)
            return
        }

        try {
            _state.value = _state.value.copy(memory = agentRepository.getMemory(), contextError = null)
        } catch (@Suppress("TooGenericExceptionCaught") e: Exception) {
            _state.value = _state.value.copy(contextError = e.toErrorMessage())
        }
    }

    private fun currentMessages(): List<AgentMessage> =
        (_state.value.history as? AgentHistoryUiState.Loaded)?.messages.orEmpty()

    /** HttpException.message is only "HTTP 400 Bad Request"; the reason is in the body. */
    private fun Exception.toErrorMessage(): String {
        if (this is HttpException) {
            val body = response()?.errorBody()?.string()
            if (!body.isNullOrBlank()) return body
        }
        return message.orEmpty()
    }
}
