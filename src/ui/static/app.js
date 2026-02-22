// Global state
let currentRun = null;
let events = [];
let tags = [];
let timeline = null;
let tracks = [];
let tracksByFrame = {};
let playerReels = [];
let playerReelsSummaryData = {};
let crossMatchReportData = null;
let crossMatchArtifacts = {};
let identityReviewData = {
    video_id: null,
    players: [],
    assignments: [],
    summary: {}
};
let recomputePreviewData = null;
let identityEditsData = [];
let identitySuggestionsData = [];
let assignmentFilterTerm = '';
let selectedAssignmentTrackIds = new Set();
let selectedSuggestionTrackIds = new Set();
let selectedReelPlayerIds = new Set();
let videoMetadata = null;
let showingOriginal = false;
let activeSegmentEndTime = null;
let playerReelFilters = {
    team: 'all',
    minScore: 0.0,
    topN: 8,
    sortBy: 'best_score_desc'
};
let pipelineJobs = [];
let pipelineConfigs = [];
let pipelineJobsPollHandle = null;
let pipelineJobsSnapshot = '';
let currentPlaybackSpeed = 1;
let speedrunState = {
    enabled: false,
    windows: [],
    lowActionWindows: [],
    durationSeconds: 0,
    eventsConsidered: 0,
    fallbackFullMatchWindow: false,
    loadedRun: null,
    currentWindowIndex: -1,
    statusMessage: 'Speedrun unavailable'
};
let viewerLayoutMode = 'split';
let speedrunJumpInProgress = false;
let eventFilterMode = 'all';
let tagFilters = {
    label: '',
    category: '',
    source: 'all'
};
let tagsLoadError = '';
let allRunsData = [];
let loadRunGeneration = 0;
let lastHighlightedEventIdx = -1;
let teamAnalyticsData = null;
let matchStatsData = null;
let runSummaryData = null;
let visualizationData = null;
let visualizationError = '';
let visualizationState = {
    type: 'pass_map',
    teamId: '',
    playerId: '',
    minConfidence: 0.0,
    includePoints: false
};
let eventPage = 0;
let assignmentPage = 0;
let suggestionPage = 0;
const EVENT_PAGE_SIZE = 20;
const ASSIGNMENT_PAGE_SIZE = 30;
const SUGGESTION_PAGE_SIZE = 30;
const VIEWER_LAYOUT_MODES = ['split', 'stacked'];
let confirmModalCallback = null;
let teamsData = [];
let currentTeamId = null;
let timelineDragging = false;

// Overlay settings
let overlaySettings = {
    layers: {
        boxes: true,
        trails: true,
        labels: true
    },
    colors: {
        team_A: '#0000ff',
        team_B: '#ff0000',
        ball: '#ffff00'
    },
    trailLength: 30
};

// DOM elements
const runsList = document.getElementById('runsList');
const viewer = document.getElementById('viewer');
const videoPlayer = document.getElementById('videoPlayer');
const videoSource = document.getElementById('videoSource');
const eventsList = document.getElementById('eventsList');
const tagsList = document.getElementById('tagsList');
const scoreDisplay = document.getElementById('scoreDisplay');
const timelineProgress = document.getElementById('timelineProgress');
const timelineBar = document.getElementById('timelineBar');
const speedrunToggleBtn = document.getElementById('toggleSpeedrunBtn');
const speedrunStatus = document.getElementById('speedrunStatus');
const layoutToggleBtn = document.getElementById('layoutToggleBtn');
const overlayCanvas = document.getElementById('overlayCanvas');
const playerReelsList = document.getElementById('playerReelsList');
const playerReelsSummary = document.getElementById('playerReelsSummary');
const reelTeamFilter = document.getElementById('reelTeamFilter');
const reelSortBy = document.getElementById('reelSortBy');
const reelMinScore = document.getElementById('reelMinScore');
const reelTopN = document.getElementById('reelTopN');
const reelSelectedCount = document.getElementById('reelSelectedCount');
const reelExportIncludeClips = document.getElementById('reelExportIncludeClips');
const seasonSummaryStatus = document.getElementById('seasonSummaryStatus');
const seasonSummaryGrid = document.getElementById('seasonSummaryGrid');
const seasonTeamTrends = document.getElementById('seasonTeamTrends');
const seasonTopPlayers = document.getElementById('seasonTopPlayers');
const seasonRecentWindow = document.getElementById('seasonRecentWindow');
const playerReelsMount = document.getElementById('playerReelsMount');
const seasonTrendsMount = document.getElementById('seasonTrendsMount');
const playerReelsRunContext = document.getElementById('playerReelsRunContext');
const seasonRunContext = document.getElementById('seasonRunContext');
const visualizationTypeSelect = document.getElementById('visualizationType');
const visualizationTeamFilter = document.getElementById('visualizationTeamFilter');
const visualizationPlayerFilter = document.getElementById('visualizationPlayerFilter');
const visualizationMinConfidence = document.getElementById('visualizationMinConfidence');
const visualizationIncludePoints = document.getElementById('visualizationIncludePoints');
const visualizationContent = document.getElementById('visualizationContent');
const crossMatchIncludeTemplates = document.getElementById('crossMatchIncludeTemplates');
const clipModal = document.getElementById('clipModal');
const clipModalTitle = document.getElementById('clipModalTitle');
const clipPlayer = document.getElementById('clipPlayer');
const identitySummary = document.getElementById('identitySummary');
const identityStatus = document.getElementById('identityStatus');
const recomputePreview = document.getElementById('recomputePreview');
const identitySuggestionsList = document.getElementById('identitySuggestionsList');
const suggestionMinConfidence = document.getElementById('suggestionMinConfidence');
const preserveExistingClipsToggle = document.getElementById('preserveExistingClips');
const suggestionsSelectedCount = document.getElementById('suggestionsSelectedCount');
const identityEditsList = document.getElementById('identityEditsList');
const identityPlayersList = document.getElementById('identityPlayersList');
const identityAssignmentsList = document.getElementById('identityAssignmentsList');
const assignmentSearch = document.getElementById('assignmentSearch');
const newPlayerName = document.getElementById('newPlayerName');
const newPlayerJersey = document.getElementById('newPlayerJersey');
const newPlayerTeam = document.getElementById('newPlayerTeam');
const mergeKeepPlayer = document.getElementById('mergeKeepPlayer');
const mergeFromPlayer = document.getElementById('mergeFromPlayer');
const bulkAssignPlayer = document.getElementById('bulkAssignPlayer');
const bulkSelectedCount = document.getElementById('bulkSelectedCount');
const pipelineVideoPaths = document.getElementById('pipelineVideoPaths');
const pipelineVideoUploadInput = document.getElementById('pipelineVideoUploadInput');
const uploadVideosBtn = document.getElementById('uploadVideosBtn');
const browseVideosBtn = document.getElementById('browseVideosBtn');
const pipelineConfigPath = document.getElementById('pipelineConfigPath');
const pipelineRunPrefix = document.getElementById('pipelineRunPrefix');
const pipelineResume = document.getElementById('pipelineResume');
const pipelineNoOverlay = document.getElementById('pipelineNoOverlay');
const pipelineSubmitBtn = document.getElementById('pipelineSubmitBtn');
const pipelineStudioStatus = document.getElementById('pipelineStudioStatus');
const pipelineJobsList = document.getElementById('pipelineJobsList');
const tagLabelInput = document.getElementById('tagLabelInput');
const tagCategoryInput = document.getElementById('tagCategoryInput');
const tagTrackInput = document.getElementById('tagTrackInput');
const tagConfidenceInput = document.getElementById('tagConfidenceInput');
const tagNotesInput = document.getElementById('tagNotesInput');
const tagFilterLabel = document.getElementById('tagFilterLabel');
const tagFilterCategory = document.getElementById('tagFilterCategory');
const tagFilterSource = document.getElementById('tagFilterSource');
const ctx = overlayCanvas.getContext('2d');
let pipelineUploadPromise = null;

// --- HLS Streaming Support ---
// Active HLS.js instance (for cleanup when switching videos)
let currentHlsInstance = null;

/**
 * Load a video source with HLS support.
 * Tries HLS playlist first (instant seeking), falls back to plain MP4.
 *
 * @param {string} runName - The run directory name
 * @param {boolean} useOverlay - If true, load overlay; if false, load original
 * @param {number|null} seekTime - Optional time to seek to after loading
 */
function loadVideoSource(runName, useOverlay = false, seekTime = null) {
    // Cleanup previous HLS instance
    if (currentHlsInstance) {
        currentHlsInstance.destroy();
        currentHlsInstance = null;
    }

    const hlsUrl = `/api/runs/${runName}/hls/playlist.m3u8`;
    const mp4Url = `/api/runs/${runName}/video` + (useOverlay ? '' : '?original=true');

    // For original video, always use MP4 (HLS is only for overlay)
    if (!useOverlay) {
        videoSource.src = mp4Url;
        videoPlayer.load();
        if (seekTime !== null) videoPlayer.currentTime = seekTime;
        return;
    }

    // Try HLS for overlay video
    if (typeof Hls !== 'undefined' && Hls.isSupported()) {
        // Check if HLS playlist exists (use GET — HEAD not supported by FileResponse)
        fetch(hlsUrl).then(resp => {
            if (resp.ok) {
                const hls = new Hls({
                    maxBufferLength: 30,
                    maxMaxBufferLength: 60,
                });
                hls.loadSource(hlsUrl);
                hls.attachMedia(videoPlayer);
                hls.on(Hls.Events.MANIFEST_PARSED, () => {
                    if (seekTime !== null) videoPlayer.currentTime = seekTime;
                });
                hls.on(Hls.Events.ERROR, (event, data) => {
                    if (data.fatal) {
                        console.warn('HLS fatal error, falling back to MP4:', data.type);
                        hls.destroy();
                        currentHlsInstance = null;
                        videoSource.src = mp4Url;
                        videoPlayer.load();
                        if (seekTime !== null) videoPlayer.currentTime = seekTime;
                    }
                });
                currentHlsInstance = hls;
            } else {
                // No HLS available, fallback to MP4
                videoSource.src = mp4Url;
                videoPlayer.load();
                if (seekTime !== null) videoPlayer.currentTime = seekTime;
            }
        }).catch(() => {
            // Network error checking HLS, fallback to MP4
            videoSource.src = mp4Url;
            videoPlayer.load();
            if (seekTime !== null) videoPlayer.currentTime = seekTime;
        });
    } else if (typeof Hls !== 'undefined' && videoPlayer.canPlayType('application/vnd.apple.mpegurl')) {
        // Safari native HLS support
        fetch(hlsUrl).then(resp => {
            if (resp.ok) {
                videoPlayer.src = hlsUrl;
                if (seekTime !== null) {
                    videoPlayer.addEventListener('loadedmetadata', () => {
                        videoPlayer.currentTime = seekTime;
                    }, { once: true });
                }
            } else {
                videoSource.src = mp4Url;
                videoPlayer.load();
                if (seekTime !== null) videoPlayer.currentTime = seekTime;
            }
        }).catch(() => {
            videoSource.src = mp4Url;
            videoPlayer.load();
            if (seekTime !== null) videoPlayer.currentTime = seekTime;
        });
    } else {
        // No HLS support, use MP4
        videoSource.src = mp4Url;
        videoPlayer.load();
        if (seekTime !== null) videoPlayer.currentTime = seekTime;
    }
}

// Restore theme from localStorage immediately
(function() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.documentElement.classList.add('dark');
    }
})();

// --- Toast Notification System ---
function showToast(message, type = 'info', duration = 3500) {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('toast-exit');
        toast.addEventListener('animationend', () => toast.remove());
    }, duration);
}

// --- Loading Bar ---
let loadingBarCount = 0;
function showLoadingBar() {
    loadingBarCount++;
    const bar = document.getElementById('loadingBar');
    if (!bar) return;
    bar.classList.add('active');
    bar.style.width = '35%';
}

function hideLoadingBar() {
    loadingBarCount = Math.max(0, loadingBarCount - 1);
    if (loadingBarCount > 0) return;
    const bar = document.getElementById('loadingBar');
    if (!bar) return;
    bar.style.width = '100%';
    setTimeout(() => {
        bar.classList.remove('active');
        bar.style.width = '0%';
    }, 300);
}

// --- Fetch with retry and error toast ---
async function fetchWithRetry(url, options = {}, retries = 1) {
    for (let attempt = 0; attempt <= retries; attempt++) {
        try {
            const response = await fetch(url, options);
            return response;
        } catch (error) {
            if (attempt === retries) {
                showToast(`Network error: ${error.message}`, 'error');
                throw error;
            }
            await new Promise(r => setTimeout(r, 500));
        }
    }
}

// --- Confirm Modal (replaces confirm()) ---
function showConfirmModal(message) {
    return new Promise((resolve) => {
        const modal = document.getElementById('confirmModal');
        const msg = document.getElementById('confirmModalMessage');
        if (!modal || !msg) { resolve(false); return; }
        msg.textContent = message;
        confirmModalCallback = resolve;
        modal.style.display = 'block';
    });
}

function confirmModalResolve() {
    const modal = document.getElementById('confirmModal');
    if (modal) modal.style.display = 'none';
    if (confirmModalCallback) {
        confirmModalCallback(true);
        confirmModalCallback = null;
    }
}

function hideConfirmModal() {
    const modal = document.getElementById('confirmModal');
    if (modal) modal.style.display = 'none';
    if (confirmModalCallback) {
        confirmModalCallback(false);
        confirmModalCallback = null;
    }
}

// --- Client-side export utilities ---
function downloadJSON(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    showToast(`Downloaded ${filename}`, 'success');
}

function downloadCSV(rows, columns, filename) {
    if (!Array.isArray(rows) || rows.length === 0) {
        showToast('No data to export', 'error');
        return;
    }
    const header = columns.join(',');
    const lines = rows.map(row =>
        columns.map(col => {
            const val = row[col] ?? '';
            const str = String(val);
            return str.includes(',') || str.includes('"') || str.includes('\n')
                ? `"${str.replace(/"/g, '""')}"`
                : str;
        }).join(',')
    );
    const csv = [header, ...lines].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    showToast(`Downloaded ${filename}`, 'success');
}

function exportSectionJSON(data, filename) {
    if (!data || (Array.isArray(data) && data.length === 0)) {
        showToast('No data to export', 'error');
        return;
    }
    downloadJSON(data, filename);
}

function exportSectionCSV(data, columns, filename) {
    if (!Array.isArray(data) || data.length === 0) {
        showToast('No data to export', 'error');
        return;
    }
    downloadCSV(data, columns, filename);
}

// --- Collapsible sections ---
function toggleCollapsible(section) {
    const body = document.getElementById(`collapsible-${section}`);
    const chevron = document.getElementById(`chevron-${section}`);
    if (!body) return;
    const collapsed = !body.classList.contains('collapsed');
    body.classList.toggle('collapsed', collapsed);
    if (chevron) chevron.classList.toggle('collapsed', collapsed);
    try { localStorage.setItem(`collapsed_${section}`, collapsed ? '1' : '0'); } catch(e) {}
}

function toggleIdentityStep(step) {
    const body = document.getElementById(`collapsible-id-${step}`);
    const chevron = document.getElementById(`chevron-id-${step}`);
    if (!body) return;
    const isHidden = body.style.display === 'none';
    body.style.display = isHidden ? '' : 'none';
    if (chevron) chevron.innerHTML = isHidden ? '&#9660;' : '&#9654;';
    try { localStorage.setItem(`idstep_${step}`, isHidden ? '0' : '1'); } catch(e) {}
}

function restoreIdentityStepStates() {
    ['suggestions', 'assignments', 'tools', 'roster'].forEach(step => {
        try {
            const stored = localStorage.getItem(`idstep_${step}`);
            if (stored === '1') {
                const body = document.getElementById(`collapsible-id-${step}`);
                const chevron = document.getElementById(`chevron-id-${step}`);
                if (body) body.style.display = 'none';
                if (chevron) chevron.innerHTML = '&#9654;';
            } else if (stored === '0') {
                const body = document.getElementById(`collapsible-id-${step}`);
                const chevron = document.getElementById(`chevron-id-${step}`);
                if (body) body.style.display = '';
                if (chevron) chevron.innerHTML = '&#9660;';
            }
        } catch(e) {}
    });
}

function restoreCollapsibleStates() {
    ['events', 'analytics', 'reels', 'season', 'identity'].forEach(section => {
        try {
            const stored = localStorage.getItem(`collapsed_${section}`);
            if (stored === '1') {
                const body = document.getElementById(`collapsible-${section}`);
                const chevron = document.getElementById(`chevron-${section}`);
                if (body) body.classList.add('collapsed');
                if (chevron) chevron.classList.add('collapsed');
            }
        } catch(e) {}
    });
}

function scrollPanelSection(sectionId) {
    const target = document.getElementById(sectionId);
    if (!target) return;
    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function relocateCrossMatchSections() {
    const reelsSection = document.querySelector('.player-reels-section');
    const seasonSection = document.querySelector('.season-section');

    if (reelsSection && playerReelsMount && reelsSection.parentElement !== playerReelsMount) {
        playerReelsMount.appendChild(reelsSection);
    }
    if (seasonSection && seasonTrendsMount && seasonSection.parentElement !== seasonTrendsMount) {
        seasonTrendsMount.appendChild(seasonSection);
    }
}

function updateCrossViewContexts() {
    const runLabel = currentRun
        ? `Current run: ${currentRun}`
        : 'Select a run in Match Analysis first.';
    if (playerReelsRunContext) playerReelsRunContext.textContent = runLabel;
    if (seasonRunContext) seasonRunContext.textContent = runLabel;
}

// --- Fullscreen ---
function toggleFullscreen() {
    const container = document.getElementById('videoContainer');
    if (!container) return;
    if (document.fullscreenElement) {
        document.exitFullscreen();
    } else {
        container.requestFullscreen().catch(() => {});
    }
}

document.addEventListener('fullscreenchange', () => {
    setTimeout(setupCanvas, 100);
});

function loadViewerLayoutPreference() {
    try {
        const stored = localStorage.getItem('viewer_layout_mode');
        if (stored && VIEWER_LAYOUT_MODES.includes(stored)) {
            return stored;
        }
    } catch (error) {
        // Ignore storage read errors and fall back to split.
    }
    return 'split';
}

function applyViewerLayout(mode) {
    const normalized = VIEWER_LAYOUT_MODES.includes(mode) ? mode : 'split';
    viewerLayoutMode = normalized;

    if (viewer) {
        viewer.classList.remove('layout-split', 'layout-stacked');
        viewer.classList.add(`layout-${normalized}`);
    }

    if (layoutToggleBtn) {
        layoutToggleBtn.textContent = normalized === 'stacked' ? 'Layout: Stacked' : 'Layout: Split';
    }

    try {
        localStorage.setItem('viewer_layout_mode', normalized);
    } catch (error) {
        // Ignore storage write errors.
    }

    setTimeout(setupCanvas, 30);
}

function toggleViewerLayout() {
    const idx = VIEWER_LAYOUT_MODES.indexOf(viewerLayoutMode);
    const nextMode = VIEWER_LAYOUT_MODES[(idx + 1) % VIEWER_LAYOUT_MODES.length];
    applyViewerLayout(nextMode);
}

function setSpeedrunStatus(message) {
    speedrunState.statusMessage = String(message || '');
    if (speedrunStatus) {
        speedrunStatus.textContent = speedrunState.statusMessage;
    }
}

function updateSpeedrunControls() {
    const hasWindows = Array.isArray(speedrunState.windows) && speedrunState.windows.length > 0;
    if (!hasWindows && speedrunState.enabled) {
        speedrunState.enabled = false;
        speedrunState.currentWindowIndex = -1;
    }

    if (speedrunToggleBtn) {
        speedrunToggleBtn.disabled = !hasWindows;
        speedrunToggleBtn.classList.toggle('active', speedrunState.enabled);
        speedrunToggleBtn.textContent = speedrunState.enabled ? 'Speedrun On' : 'Speedrun Off';
    }

    if (!speedrunState.statusMessage) {
        if (!hasWindows) {
            speedrunState.statusMessage = 'Speedrun unavailable';
        } else if (speedrunState.fallbackFullMatchWindow) {
            speedrunState.statusMessage = 'Speedrun ready (full match)';
        } else {
            speedrunState.statusMessage = `Speedrun ready (${speedrunState.windows.length} windows)`;
        }
    }

    if (speedrunStatus) {
        speedrunStatus.textContent = speedrunState.statusMessage;
    }
}

function resetSpeedrunState() {
    speedrunState.enabled = false;
    speedrunState.windows = [];
    speedrunState.lowActionWindows = [];
    speedrunState.durationSeconds = 0;
    speedrunState.eventsConsidered = 0;
    speedrunState.fallbackFullMatchWindow = false;
    speedrunState.loadedRun = null;
    speedrunState.currentWindowIndex = -1;
    speedrunState.statusMessage = 'Speedrun unavailable';
    speedrunJumpInProgress = false;
    updateSpeedrunControls();
}

function normalizeSpeedrunWindow(windowRow) {
    const start = Number(windowRow && windowRow.start);
    const end = Number(windowRow && windowRow.end);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
        return null;
    }
    return {
        start: Math.max(0, start),
        end: Math.max(start, end),
        duration: Math.max(0, end - start),
        eventCount: Number(windowRow && windowRow.event_count) || 0
    };
}

async function loadSpeedrunWindows(runName) {
    if (!runName) return;

    speedrunState.enabled = false;
    speedrunState.loadedRun = runName;
    speedrunState.currentWindowIndex = -1;
    speedrunState.statusMessage = 'Loading speedrun...';
    updateSpeedrunControls();

    try {
        const response = await fetchWithRetry(`/api/runs/${runName}/playback/speedrun`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const payload = await response.json();
        const rawWindows = Array.isArray(payload.high_action_windows)
            ? payload.high_action_windows
            : [];
        const rawLowAction = Array.isArray(payload.low_action_windows)
            ? payload.low_action_windows
            : [];

        speedrunState.windows = rawWindows
            .map(normalizeSpeedrunWindow)
            .filter(Boolean)
            .sort((a, b) => a.start - b.start);
        speedrunState.lowActionWindows = rawLowAction
            .map(normalizeSpeedrunWindow)
            .filter(Boolean)
            .sort((a, b) => a.start - b.start);
        speedrunState.durationSeconds = Number(payload.duration_seconds) || 0;
        speedrunState.eventsConsidered = Number(payload.events_considered) || 0;
        speedrunState.fallbackFullMatchWindow = Boolean(payload.fallback_full_match_window);
        speedrunState.currentWindowIndex = -1;

        if (speedrunState.windows.length === 0) {
            setSpeedrunStatus('Speedrun unavailable');
        } else if (speedrunState.fallbackFullMatchWindow) {
            setSpeedrunStatus('Speedrun ready (full match)');
        } else {
            setSpeedrunStatus(`Speedrun ready (${speedrunState.windows.length} windows)`);
        }
        updateSpeedrunControls();
    } catch (error) {
        console.error('Error loading speedrun windows:', error);
        speedrunState.windows = [];
        speedrunState.lowActionWindows = [];
        speedrunState.durationSeconds = Number(videoPlayer.duration || 0);
        speedrunState.eventsConsidered = 0;
        speedrunState.fallbackFullMatchWindow = false;
        speedrunState.currentWindowIndex = -1;
        setSpeedrunStatus('Speedrun unavailable');
        updateSpeedrunControls();
    }
}

function findSpeedrunWindowIndex(timeValue) {
    if (!Array.isArray(speedrunState.windows) || speedrunState.windows.length === 0) {
        return -1;
    }
    return speedrunState.windows.findIndex((windowRow) => (
        timeValue >= windowRow.start && timeValue <= windowRow.end
    ));
}

function findNextSpeedrunWindowIndex(timeValue) {
    if (!Array.isArray(speedrunState.windows) || speedrunState.windows.length === 0) {
        return -1;
    }
    return speedrunState.windows.findIndex((windowRow) => timeValue < windowRow.start);
}

function jumpToSpeedrunWindow(index, keepPlayState = true) {
    const targetWindow = speedrunState.windows[index];
    if (!targetWindow) return false;

    const shouldPlay = keepPlayState ? !videoPlayer.paused : false;
    speedrunJumpInProgress = true;
    speedrunState.currentWindowIndex = index;

    if (shouldPlay) {
        _seekAndPlay(targetWindow.start);
    } else {
        videoPlayer.currentTime = targetWindow.start;
    }
    speedrunJumpInProgress = false;

    setSpeedrunStatus(`Speedrun active ${index + 1}/${speedrunState.windows.length}`);
    return true;
}

function disableSpeedrunMode() {
    speedrunState.enabled = false;
    speedrunState.currentWindowIndex = -1;

    if (speedrunState.windows.length === 0) {
        setSpeedrunStatus('Speedrun unavailable');
    } else if (speedrunState.fallbackFullMatchWindow) {
        setSpeedrunStatus('Speedrun ready (full match)');
    } else {
        setSpeedrunStatus(`Speedrun ready (${speedrunState.windows.length} windows)`);
    }
    updateSpeedrunControls();
}

function toggleSpeedrunMode() {
    if (!speedrunState.windows.length) {
        showToast('Speedrun windows are unavailable for this run.', 'error');
        return;
    }

    if (speedrunState.enabled) {
        disableSpeedrunMode();
        return;
    }

    speedrunState.enabled = true;
    speedrunState.currentWindowIndex = -1;
    updateSpeedrunControls();

    const now = Number(videoPlayer.currentTime || 0);
    const activeIndex = findSpeedrunWindowIndex(now);
    if (activeIndex >= 0) {
        speedrunState.currentWindowIndex = activeIndex;
        setSpeedrunStatus(`Speedrun active ${activeIndex + 1}/${speedrunState.windows.length}`);
        return;
    }

    const nextIndex = findNextSpeedrunWindowIndex(now);
    if (nextIndex >= 0) {
        jumpToSpeedrunWindow(nextIndex, true);
        return;
    }

    jumpToSpeedrunWindow(0, true);
}

function enforceSpeedrunPlayback() {
    if (!speedrunState.enabled || !speedrunState.windows.length || speedrunJumpInProgress) {
        return;
    }

    const now = Number(videoPlayer.currentTime || 0);
    const activeIndex = findSpeedrunWindowIndex(now);
    if (activeIndex >= 0) {
        if (speedrunState.currentWindowIndex !== activeIndex) {
            speedrunState.currentWindowIndex = activeIndex;
            setSpeedrunStatus(`Speedrun active ${activeIndex + 1}/${speedrunState.windows.length}`);
        }

        const activeWindow = speedrunState.windows[activeIndex];
        if (now >= (activeWindow.end - 0.02)) {
            const nextIndex = activeIndex + 1;
            if (nextIndex < speedrunState.windows.length) {
                jumpToSpeedrunWindow(nextIndex, true);
            } else {
                videoPlayer.pause();
                disableSpeedrunMode();
                setSpeedrunStatus('Speedrun complete');
            }
        }
        return;
    }

    const nextIndex = findNextSpeedrunWindowIndex(now);
    if (nextIndex >= 0) {
        jumpToSpeedrunWindow(nextIndex, true);
    } else {
        videoPlayer.pause();
        disableSpeedrunMode();
        setSpeedrunStatus('Speedrun complete');
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Update theme toggle button text
    const isDark = document.documentElement.classList.contains('dark');
    const themeBtn = document.getElementById('themeToggle');
    if (themeBtn) {
        themeBtn.textContent = isDark ? 'Light Mode' : 'Dark Mode';
    }

    relocateCrossMatchSections();
    restoreCollapsibleStates();
    restoreIdentityStepStates();
    applyViewerLayout(loadViewerLayoutPreference());
    updateSpeedrunControls();
    updateCrossViewContexts();
    updatePlayerReelFilters();
    _syncTagFilterInputs();
    if (tagCategoryInput && !String(tagCategoryInput.value || '').trim()) {
        tagCategoryInput.value = 'general';
    }
    loadPipelineConfigs();
    loadPipelineJobs(true);
    loadPipelineTeamSelectors();
    initPipelineUploadControls();
    restoreFromHash();
    setupVideoPlayer();
    if (pipelineJobsPollHandle == null) {
        pipelineJobsPollHandle = setInterval(() => {
            loadPipelineJobs(false);
        }, 3000);
    }
});

function setPipelineStatus(message, isError = false) {
    if (!pipelineStudioStatus) return;
    pipelineStudioStatus.textContent = message || '';
    pipelineStudioStatus.classList.toggle('status-error', isError);
    pipelineStudioStatus.classList.toggle('status-info', !isError);
    if (isError && message) showToast(message, 'error');
}

function isMobileBrowser() {
    const ua = navigator.userAgent || '';
    const touchMac = navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1;
    return /Android|webOS|iPhone|iPad|iPod|IEMobile|Opera Mini/i.test(ua) || touchMac;
}

function setPipelineUploadBusy(isBusy) {
    if (uploadVideosBtn) uploadVideosBtn.disabled = isBusy;
    if (browseVideosBtn) browseVideosBtn.disabled = isBusy;
    if (pipelineSubmitBtn) pipelineSubmitBtn.disabled = isBusy;
}

function appendPipelineVideoPaths(paths) {
    if (!pipelineVideoPaths || !Array.isArray(paths) || paths.length === 0) return;
    const existingLines = pipelineVideoPaths.value
        .split('\n')
        .map((line) => line.trim())
        .filter((line) => line.length > 0);
    const nextLines = paths
        .map((path) => String(path || '').trim())
        .filter((path) => path.length > 0);
    if (nextLines.length === 0) return;

    pipelineVideoPaths.value = existingLines.concat(nextLines).join('\n');
    pipelineVideoPaths.setCustomValidity('');
    pipelineVideoPaths.dispatchEvent(new Event('input', { bubbles: true }));
    pipelineVideoPaths.dispatchEvent(new Event('change', { bubbles: true }));
}

function triggerVideoUploadPicker() {
    if (pipelineVideoUploadInput) {
        pipelineVideoUploadInput.click();
    }
}

function initPipelineUploadControls() {
    if (pipelineVideoUploadInput) {
        pipelineVideoUploadInput.addEventListener('change', handlePipelineVideoUploadSelection);
    }
    if (browseVideosBtn && isMobileBrowser()) {
        browseVideosBtn.style.display = 'none';
    }
}

async function uploadPipelineVideoFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/api/upload-video', {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
            const errorData = await response.json();
            if (errorData && errorData.detail) detail = errorData.detail;
        } catch (parseError) {
            // Keep fallback detail from status.
        }
        throw new Error(detail);
    }

    const payload = await response.json();
    if (!payload || !payload.path) {
        throw new Error('Upload succeeded but no path was returned');
    }
    return payload.path;
}

async function handlePipelineVideoUploadSelection(evt) {
    const inputEl = evt && evt.target ? evt.target : pipelineVideoUploadInput;
    if (!inputEl || !inputEl.files || inputEl.files.length === 0) return;
    if (pipelineUploadPromise) {
        setPipelineStatus('A video upload is already in progress. Please wait...');
        inputEl.value = '';
        return;
    }

    const selectedFiles = Array.from(inputEl.files);
    setPipelineUploadBusy(true);

    const uploadTask = (async () => {
        setPipelineStatus(`Uploading ${selectedFiles.length} video file${selectedFiles.length === 1 ? '' : 's'}...`);
        const uploadedPaths = [];
        for (const file of selectedFiles) {
            const uploadedPath = await uploadPipelineVideoFile(file);
            uploadedPaths.push(uploadedPath);
        }
        appendPipelineVideoPaths(uploadedPaths);
        setPipelineStatus(`Uploaded ${uploadedPaths.length} video file${uploadedPaths.length === 1 ? '' : 's'}.`);
    })();
    pipelineUploadPromise = uploadTask;

    try {
        await uploadTask;
    } catch (error) {
        console.error('Error uploading videos:', error);
        setPipelineStatus(`Failed to upload video: ${error.message}`, true);
    } finally {
        inputEl.value = '';
        if (pipelineUploadPromise === uploadTask) {
            pipelineUploadPromise = null;
        }
        setPipelineUploadBusy(false);
    }
}

function pipelineConfigValue() {
    if (!pipelineConfigPath) return null;
    const value = pipelineConfigPath.value;
    if (!value || value === '__builtin_default__') {
        return null;
    }
    return value;
}

async function loadPipelineConfigs() {
    if (!pipelineConfigPath) return;

    try {
        const response = await fetch('/api/pipeline/configs');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        pipelineConfigs = Array.isArray(data.configs) ? data.configs : [];

        let html = '';
        pipelineConfigs.forEach((cfg) => {
            const optionValue = cfg.path ? cfg.path : '__builtin_default__';
            const label = cfg.path
                ? `${cfg.label} (${cfg.path})`
                : `${cfg.label} (recommended for first run)`;
            html += `<option value="${escapeHtml(optionValue)}">${escapeHtml(label)}</option>`;
        });

        if (!html) {
            html = '<option value="__builtin_default__">Built-in default</option>';
        }
        pipelineConfigPath.innerHTML = html;
        updateConfigDescription();
    } catch (error) {
        console.error('Error loading pipeline configs:', error);
        pipelineConfigPath.innerHTML = '<option value="__builtin_default__">Built-in default</option>';
    }
}

function updateConfigDescription() {
    const descDiv = document.getElementById('configDescription');
    const descText = document.getElementById('configDescText');
    const estTime = document.getElementById('configEstTime');
    const estSize = document.getElementById('configEstSize');
    if (!descDiv || !pipelineConfigPath) return;

    const selectedValue = pipelineConfigPath.value;
    const cfg = pipelineConfigs.find(c =>
        (c.path || '__builtin_default__') === selectedValue
    );
    if (!cfg || (!cfg.description && !cfg.estimate_time)) {
        descDiv.style.display = 'none';
        return;
    }
    descDiv.style.display = '';
    descText.textContent = cfg.description || '';
    estTime.textContent = cfg.estimate_time || '';
    estTime.style.display = cfg.estimate_time ? '' : 'none';
    estSize.textContent = cfg.estimate_size || '';
    estSize.style.display = cfg.estimate_size ? '' : 'none';
}

function pipelineProgressPercent(job) {
    const stageIndex = Number(job.stage_index || 0);
    const stageTotal = Number(job.stage_total || 0);
    if (!Number.isFinite(stageIndex) || !Number.isFinite(stageTotal) || stageTotal <= 0) {
        return 0;
    }
    return Math.max(0, Math.min(100, Math.round((stageIndex / stageTotal) * 100)));
}

function formatPipelineTime(value) {
    if (!value) return 'n/a';
    try {
        return new Date(value).toLocaleString();
    } catch (error) {
        return String(value);
    }
}

function renderPipelineJobs() {
    if (!pipelineJobsList) return;

    if (!Array.isArray(pipelineJobs) || pipelineJobs.length === 0) {
        pipelineJobsList.innerHTML = '<p class="loading">No jobs queued yet.</p>';
        return;
    }

    let html = '';
    pipelineJobs.forEach((job) => {
        const status = String(job.status || 'queued');
        const stageName = job.stage_name ? escapeHtml(String(job.stage_name)) : 'waiting';
        const stageIndex = Number(job.stage_index || 0);
        const stageTotal = Number(job.stage_total || 0);
        const progress = pipelineProgressPercent(job);
        const stageLabel = stageTotal > 0 ? `${stageIndex}/${stageTotal}` : '0/0';
        const outputDir = escapeHtml(String(job.output_dir || ''));
        const runName = escapeHtml(String(job.run_name || ''));
        const encodedRunName = encodeURIComponent(String(job.run_name || ''));
        const jobId = String(job.job_id || '');
        const cancelledPending = Boolean(job.cancel_requested);
        const message = job.message ? escapeHtml(String(job.message)) : '';
        const errorText = job.error ? `<div class="pipeline-job-meta status-error">${escapeHtml(String(job.error))}</div>` : '';
        const openRunAction = status === 'succeeded' && job.run_name
            ? `<button class="identity-btn" onclick="showView('matchAnalysisView');loadRunByEncodedName('${encodedRunName}')">Open Run</button>`
            : '';
        const cancelAction = (status === 'queued' || status === 'running')
            ? `<button class="identity-btn" ${cancelledPending ? 'disabled' : ''} onclick="cancelPipelineJob('${jobId}')">${cancelledPending ? 'Cancel Requested' : 'Cancel'}</button>`
            : '';
        const resumeAction = (status === 'failed' || status === 'cancelled')
            ? `<button class="identity-btn" onclick="resumePipelineJob('${jobId}')">Resume</button>`
            : '';
        const retryAction = (status === 'failed' || status === 'cancelled')
            ? `<button class="identity-btn" onclick="retryPipelineJob('${jobId}')">Retry</button>`
            : '';
        const duplicateAction = (status === 'succeeded' || status === 'failed' || status === 'cancelled')
            ? `<button class="identity-btn" onclick="duplicatePipelineJob('${jobId}')">Duplicate</button>`
            : '';
        const deleteAction = (status === 'failed' || status === 'cancelled' || status === 'queued' || status === 'succeeded')
            ? `<button class="identity-btn" style="color:#c33" onclick="deletePipelineJob('${jobId}')">Delete</button>`
            : '';
        const sourceTag = job.source_job_id
            ? `<div class="pipeline-job-meta">From job: ${escapeHtml(String(job.source_job_id))}</div>`
            : '';

        html += `
            <div class="pipeline-job-card">
                <div class="pipeline-job-top">
                    <div class="pipeline-job-title">${escapeHtml(String(job.video_name || job.video_path || 'Video'))}</div>
                    <span class="pipeline-status-badge ${escapeHtml(status)}">${escapeHtml(status)}</span>
                </div>
                <div class="pipeline-job-meta">Run: <strong>${runName}</strong></div>
                <div class="pipeline-job-meta">Stage: ${escapeHtml(stageLabel)} • ${stageName}</div>
                <div class="pipeline-job-meta">Output: ${outputDir}</div>
                <div class="pipeline-job-meta">Created: ${escapeHtml(formatPipelineTime(job.created_at))}</div>
                ${sourceTag}
                ${message ? `<div class="pipeline-job-meta">${message}</div>` : ''}
                ${errorText}
                <div class="pipeline-progress"><div class="pipeline-progress-fill" style="width:${progress}%;"></div></div>
                <div class="pipeline-job-actions">
                    ${cancelAction}
                    ${resumeAction}
                    ${retryAction}
                    ${duplicateAction}
                    ${openRunAction}
                    ${deleteAction}
                </div>
            </div>
        `;
    });
    pipelineJobsList.innerHTML = html;
}

async function loadPipelineJobs(forceRender = false) {
    const previousStatuses = new Map(
        (pipelineJobs || []).map((job) => [job.job_id, job.status])
    );

    try {
        const response = await fetch('/api/pipeline/jobs?limit=80');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        const nextJobs = Array.isArray(data.jobs) ? data.jobs : [];
        const maxParallelJobs = Number(data.max_parallel_jobs || 1);
        const nextSnapshot = JSON.stringify(
            nextJobs.map((job) => [
                job.job_id,
                job.status,
                job.stage_name,
                job.stage_index,
                job.stage_total,
                job.message,
                job.error,
                job.cancel_requested
            ])
        );
        pipelineJobs = nextJobs;

        const runningCount = nextJobs.filter((job) => job.status === 'running').length;
        const queuedCount = nextJobs.filter((job) => job.status === 'queued').length;
        if (runningCount > 0 || queuedCount > 0) {
            setPipelineStatus(`${runningCount} running • ${queuedCount} queued • concurrency ${maxParallelJobs}`);
        } else if (nextJobs.length > 0) {
            setPipelineStatus(`No active jobs • concurrency ${maxParallelJobs}`);
        } else {
            setPipelineStatus(`Ready • concurrency ${maxParallelJobs}`);
        }

        if (forceRender || pipelineJobsSnapshot !== nextSnapshot) {
            pipelineJobsSnapshot = nextSnapshot;
            renderPipelineJobs();
        }

        const hasNewSuccess = nextJobs.some((job) => {
            if (job.status !== 'succeeded') return false;
            return previousStatuses.get(job.job_id) !== 'succeeded';
        });
        if (hasNewSuccess) {
            loadRuns();
        }
    } catch (error) {
        console.error('Error loading pipeline jobs:', error);
        if (forceRender && pipelineJobsList) {
            pipelineJobsList.innerHTML = '<p class="loading">Unable to load jobs right now.</p>';
        }
    }
}

async function postPipelineJobAction(jobId, actionPath, statusPrefix) {
    try {
        const response = await fetch(`/api/pipeline/jobs/${encodeURIComponent(jobId)}/${actionPath}`, {
            method: 'POST'
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }
        setPipelineStatus(statusPrefix);
        await loadPipelineJobs(true);
        await loadRuns();
    } catch (error) {
        console.error(`Error in job action ${actionPath}:`, error);
        setPipelineStatus(`Action failed: ${error.message}`, true);
    }
}

async function cancelPipelineJob(jobId) {
    await postPipelineJobAction(jobId, 'cancel', 'Cancellation request sent.');
}

async function retryPipelineJob(jobId) {
    await postPipelineJobAction(jobId, 'retry', 'Retry job queued.');
}

async function resumePipelineJob(jobId) {
    await postPipelineJobAction(jobId, 'resume', 'Resume job queued.');
}

async function duplicatePipelineJob(jobId) {
    await postPipelineJobAction(jobId, 'duplicate', 'Duplicate job queued.');
}

async function deletePipelineJob(jobId) {
    try {
        const response = await fetch(`/api/pipeline/jobs/${encodeURIComponent(jobId)}?clean_files=true`, {
            method: 'DELETE'
        });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }
        setPipelineStatus('Job deleted.');
        await loadPipelineJobs(true);
        await loadRuns();
    } catch (error) {
        console.error('Error deleting job:', error);
        setPipelineStatus(`Delete failed: ${error.message}`, true);
    }
}

async function browseVideos() {
    if (isMobileBrowser()) {
        triggerVideoUploadPicker();
        return;
    }
    const btn = browseVideosBtn;
    if (btn) btn.disabled = true;
    try {
        const response = await fetch('/api/browse-videos', { method: 'POST' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        const paths = data.paths || [];
        appendPipelineVideoPaths(paths);
    } catch (error) {
        console.error('Error browsing videos:', error);
        setPipelineStatus('Failed to open file browser: ' + error.message, true);
    } finally {
        if (btn) btn.disabled = false;
    }
}

async function submitPipelineJobs(formEvent) {
    formEvent.preventDefault();

    if (!pipelineVideoPaths) return;
    if (pipelineUploadPromise) {
        setPipelineStatus('Waiting for upload to finish...');
        try {
            await pipelineUploadPromise;
        } catch (error) {
            setPipelineStatus(`Failed to queue jobs: ${error.message}`, true);
            return;
        }
    }

    const lines = pipelineVideoPaths.value
        .split('\n')
        .map((line) => line.trim())
        .filter((line) => line.length > 0);

    if (lines.length === 0) {
        setPipelineStatus('Enter at least one video path.', true);
        return;
    }

    const payload = {
        video_paths: lines,
        resume: pipelineResume ? Boolean(pipelineResume.checked) : false,
        no_overlay: pipelineNoOverlay ? Boolean(pipelineNoOverlay.checked) : false
    };

    // Add team pre-selection if set
    const homeTeamSel = document.getElementById('pipelineHomeTeam');
    const awayTeamSel = document.getElementById('pipelineAwayTeam');
    const homeKitSel = document.getElementById('pipelineHomeKit');
    const awayKitSel = document.getElementById('pipelineAwayKit');
    if (homeTeamSel && homeTeamSel.value) payload.home_team_id = parseInt(homeTeamSel.value);
    if (awayTeamSel && awayTeamSel.value) payload.away_team_id = parseInt(awayTeamSel.value);
    if (homeKitSel) payload.home_kit = homeKitSel.value;
    if (awayKitSel) payload.away_kit = awayKitSel.value;

    const configPathValue = pipelineConfigValue();
    if (configPathValue) {
        payload.config_path = configPathValue;
    }

    const runPrefixValue = pipelineRunPrefix ? pipelineRunPrefix.value.trim() : '';
    if (runPrefixValue) {
        if (lines.length === 1) {
            payload.run_name = runPrefixValue;
        } else {
            payload.run_name_prefix = runPrefixValue;
        }
    }

    if (pipelineSubmitBtn) {
        pipelineSubmitBtn.disabled = true;
    }
    setPipelineStatus('Queueing pipeline job(s)...');

    try {
        const response = await fetch('/api/pipeline/jobs', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        const accepted = Number(data.accepted_count || 0);
        setPipelineStatus(`Queued ${accepted} pipeline job${accepted === 1 ? '' : 's'}.`);
        pipelineVideoPaths.value = '';
        await loadPipelineJobs(true);
    } catch (error) {
        console.error('Error queueing pipeline jobs:', error);
        setPipelineStatus(`Failed to queue jobs: ${error.message}`, true);
    } finally {
        if (pipelineSubmitBtn) {
            pipelineSubmitBtn.disabled = false;
        }
    }
}

// Load available runs
async function loadRuns() {
    try {
        const response = await fetch('/api/runs');
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        allRunsData = data.runs || [];
        renderRunsList();
    } catch (error) {
        console.error('Error loading runs:', error);
        allRunsData = [];
        const content = document.getElementById('runsListContent');
        if (content) {
            content.innerHTML = '<p class="loading">Error loading runs</p>';
        }
    }
}

function renderRunsList() {
    const content = document.getElementById('runsListContent');
    if (!content) return;

    const searchInput = document.getElementById('runSearchInput');
    const searchTerm = searchInput ? searchInput.value.trim().toLowerCase() : '';
    const sortBy = document.getElementById('runSortBy');
    const sortMode = sortBy ? sortBy.value : 'name_asc';

    const filtered = allRunsData.filter(run => {
        if (!searchTerm) return true;
        return (run.name || '').toLowerCase().includes(searchTerm);
    });

    // Sort
    filtered.sort((a, b) => {
        if (sortMode === 'name_desc') return (b.name || '').localeCompare(a.name || '');
        if (sortMode === 'events_desc') {
            const aCount = (a.event_counts?.shot || 0) + (a.event_counts?.goal || 0);
            const bCount = (b.event_counts?.shot || 0) + (b.event_counts?.goal || 0);
            return bCount - aCount;
        }
        if (sortMode === 'duration_desc') return (b.duration || 0) - (a.duration || 0);
        return (a.name || '').localeCompare(b.name || '');
    });

    // Onboarding card
    renderOnboardingCard();

    if (filtered.length === 0) {
        content.innerHTML = searchTerm
            ? '<p class="loading">No runs match your search</p>'
            : '<p class="loading">No runs yet. Queue videos from <a href="#" onclick="showView(\'pipelineStudioView\');return false;" style="color:var(--accent);text-decoration:underline;">Pipeline Studio</a>.</p>';
        return;
    }

    let html = '';
    filtered.forEach(run => {
        const duration = run.duration ? `${(run.duration / 60).toFixed(1)}min` : 'N/A';
        const resolution = run.resolution || 'N/A';
        const encodedRunName = encodeURIComponent(run.name);

        let badges = '';
        if (run.event_counts) {
            if (run.event_counts.shot > 0) {
                badges += `<span class="event-badge shot">${run.event_counts.shot} shots</span>`;
            }
            if (run.event_counts.goal > 0) {
                badges += `<span class="event-badge goal">${run.event_counts.goal} goals</span>`;
            }
        }
        if (run.player_reel_summary && run.player_reel_summary.players_with_reels > 0) {
            badges += `<span class="event-badge player">${run.player_reel_summary.players_with_reels} player reels</span>`;
        }
        if (run.cross_match_summary && Number(run.cross_match_summary.matches_analyzed || 0) > 0) {
            badges += `<span class="event-badge goal">${run.cross_match_summary.matches_analyzed} season matches</span>`;
        }

        const activeClass = currentRun === run.name ? 'active' : '';
        html += `
            <div class="run-item ${activeClass}" data-run-name="${escapeHtml(run.name)}" onclick="loadRunByEncodedName('${encodedRunName}', this)">
                <div style="display:flex;justify-content:space-between;align-items:start;">
                    <div class="run-name">${escapeHtml(run.name)}</div>
                    <button class="identity-btn" style="color:#c33;padding:2px 6px;font-size:0.75rem;flex-shrink:0;" onclick="event.stopPropagation();deleteRun('${encodedRunName}')">Delete</button>
                </div>
                <div class="run-meta">
                    ${duration} • ${resolution} • ${run.fps ? run.fps.toFixed(0) + 'fps' : 'N/A'}
                    <div style="margin-top: 0.5rem;">${badges}</div>
                </div>
            </div>
        `;
    });

    content.innerHTML = html;
}

function filterRuns() {
    renderRunsList();
}

function loadRunByEncodedName(encodedRunName, sourceElement = null) {
    const runName = decodeURIComponent(encodedRunName);
    return loadRun(runName, sourceElement);
}

async function deleteRun(encodedRunName) {
    const runName = decodeURIComponent(encodedRunName);
    try {
        const response = await fetch(`/api/runs/${encodedRunName}`, { method: 'DELETE' });
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }
        showToast(`Run "${runName}" deleted.`);
        if (currentRun === runName) {
            currentRun = null;
            updateCrossViewContexts();
        }
        await loadRuns();
        await loadPipelineJobs(true);
    } catch (error) {
        console.error('Error deleting run:', error);
        showToast(`Delete failed: ${error.message}`, 'error');
    }
}

// Load specific run
async function loadRun(runName, sourceElement = null) {
    const thisGeneration = ++loadRunGeneration;

    currentRun = runName;
    updateCrossViewContexts();

    // Reset state
    tags = [];
    tagsLoadError = '';
    tracks = [];
    tracksByFrame = {};
    playerReels = [];
    playerReelsSummaryData = {};
    crossMatchReportData = null;
    crossMatchArtifacts = {};
    identityReviewData = { video_id: null, players: [], assignments: [], summary: {} };
    recomputePreviewData = null;
    identityEditsData = [];
    identitySuggestionsData = [];
    teamAnalyticsData = null;
    matchStatsData = null;
    runSummaryData = null;
    visualizationData = null;
    visualizationError = '';
    assignmentFilterTerm = '';
    selectedAssignmentTrackIds.clear();
    selectedSuggestionTrackIds.clear();
    selectedReelPlayerIds.clear();
    lastLoadedFrame = 0;
    activeSegmentEndTime = null;
    resetSpeedrunState();
    eventFilterMode = 'all';
    tagFilters = { label: '', category: '', source: 'all' };
    _syncTagFilterInputs();
    eventPage = 0;
    assignmentPage = 0;
    suggestionPage = 0;
    lastHighlightedEventIdx = -1;
    resetEventFilterButtons();
    hideClipModal();
    renderRecomputePreview();
    renderIdentitySuggestions();
    renderIdentityEdits();
    renderCrossMatchReport();
    renderTeamAnalytics();
    renderMatchStats();
    renderRunSummary();
    renderVisualizationPanel();
    renderTags();

    // Update UI
    document.querySelectorAll('.run-item').forEach(item => item.classList.remove('active'));
    if (sourceElement && sourceElement.classList) {
        sourceElement.classList.add('active');
    } else {
        document.querySelectorAll('.run-item').forEach((item) => {
            if (item.dataset.runName === runName) {
                item.classList.add('active');
            }
        });
    }

    viewer.classList.add('active');
    applyViewerLayout(viewerLayoutMode);
    updateUrlHash();

    showLoadingBar();

    // Load video (original by default for dynamic overlay)
    loadVideoSource(runName, false);

    try {
        // Load metadata
        await loadMetadata(runName);
        if (thisGeneration !== loadRunGeneration) return;

        // Load events
        await loadEvents(runName);
        if (thisGeneration !== loadRunGeneration) return;

        // Load speedrun windows for low-action skipping mode
        await loadSpeedrunWindows(runName);
        if (thisGeneration !== loadRunGeneration) return;

        // Load run tags
        await loadTags(runName);
        if (thisGeneration !== loadRunGeneration) return;

        // Load team analytics
        await loadTeamAnalytics(runName);
        if (thisGeneration !== loadRunGeneration) return;

        // Load match stats
        await loadMatchStats(runName);
        if (thisGeneration !== loadRunGeneration) return;

        // Load tactical/pass visualization artifact
        await loadVisualization(runName);
        if (thisGeneration !== loadRunGeneration) return;

        // Load run-team mapping bar
        await loadRunTeamMapping(runName);
        if (thisGeneration !== loadRunGeneration) return;

        // Load lineup, notes, spotlight config
        loadLineup(runName);
        loadCoachNotes(runName);
        loadSpotlightConfig(runName);
        if (thisGeneration !== loadRunGeneration) return;

        // Load per-player highlight reels
        await loadPlayerReels(runName);
        if (thisGeneration !== loadRunGeneration) return;

        // Load cross-match season report
        await loadCrossMatchReport(runName);
        if (thisGeneration !== loadRunGeneration) return;

        // Load identity review data
        await loadIdentityReview(runName);
        if (thisGeneration !== loadRunGeneration) return;
        await loadIdentitySuggestions(runName);
        if (thisGeneration !== loadRunGeneration) return;
        await loadIdentityEdits(runName);
        if (thisGeneration !== loadRunGeneration) return;

        // Load timeline
        await loadTimeline(runName);
        if (thisGeneration !== loadRunGeneration) return;

        // Load tracks for overlay
        await loadTracks(runName);
    } finally {
        hideLoadingBar();
    }
}

// Load events for a run
async function loadEvents(runName) {
    try {
        const response = await fetch(`/api/runs/${runName}/events`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        events = data.events || [];

        renderEvents();
        renderTimelineMarkers();
    } catch (error) {
        console.error('Error loading events:', error);
        events = [];
        eventsList.innerHTML = '<p class="loading">No events found</p>';
    }
}

function _normalizeTagFiltersFromInputs() {
    tagFilters = {
        label: tagFilterLabel ? String(tagFilterLabel.value || '').trim() : '',
        category: tagFilterCategory ? String(tagFilterCategory.value || '').trim() : '',
        source: tagFilterSource ? String(tagFilterSource.value || 'all') : 'all'
    };
}

function _syncTagFilterInputs() {
    if (tagFilterLabel) tagFilterLabel.value = tagFilters.label || '';
    if (tagFilterCategory) tagFilterCategory.value = tagFilters.category || '';
    if (tagFilterSource) tagFilterSource.value = tagFilters.source || 'all';
}

function _tagTimeLabel(tag) {
    const start = Number(tag?.start_time);
    const end = Number(tag?.end_time);
    const hasStart = Number.isFinite(start);
    const hasEnd = Number.isFinite(end);
    if (hasStart && hasEnd && Math.abs(end - start) >= 0.01) {
        return `${formatTime(start)} - ${formatTime(end)}`;
    }
    if (hasStart) return formatTime(start);
    if (hasEnd) return formatTime(end);
    return 'n/a';
}

function _tagSeekTime(tag) {
    const start = Number(tag?.start_time);
    if (Number.isFinite(start)) return start;
    const end = Number(tag?.end_time);
    if (Number.isFinite(end)) return end;
    return null;
}

function renderTags() {
    if (!tagsList) return;
    if (!currentRun) {
        tagsList.innerHTML = '<p class="loading">Select a run to load tags</p>';
        return;
    }
    if (tagsLoadError) {
        tagsList.innerHTML = `<p class="loading">${escapeHtml(tagsLoadError)}</p>`;
        return;
    }
    if (!Array.isArray(tags) || tags.length === 0) {
        tagsList.innerHTML = '<p class="loading">No tags match current filters</p>';
        return;
    }

    const html = tags.map((tag) => {
        const tagId = Number(tag.tag_id || 0);
        const label = escapeHtml(String(tag.label || 'tag'));
        const category = escapeHtml(String(tag.category || 'general'));
        const source = escapeHtml(String(tag.source || 'manual'));
        const timeLabel = escapeHtml(_tagTimeLabel(tag));
        const seekTime = _tagSeekTime(tag);
        const seekAttr = Number.isFinite(seekTime) ? `onclick="seekToEvent(${seekTime})"` : '';

        const metaParts = [];
        if (tag.track_id != null) metaParts.push(`track ${escapeHtml(String(tag.track_id))}`);
        if (tag.player_name) {
            metaParts.push(`player ${escapeHtml(String(tag.player_name))}`);
        } else if (tag.player_id != null) {
            metaParts.push(`player #${escapeHtml(String(tag.player_id))}`);
        }
        if (tag.team_name) {
            metaParts.push(`team ${escapeHtml(String(tag.team_name))}`);
        } else if (tag.team_id != null) {
            metaParts.push(`team #${escapeHtml(String(tag.team_id))}`);
        }
        if (tag.confidence != null && Number.isFinite(Number(tag.confidence))) {
            metaParts.push(`conf ${Number(tag.confidence).toFixed(2)}`);
        }
        if (tag.notes) {
            metaParts.push(`note ${escapeHtml(String(tag.notes))}`);
        }
        const metaText = metaParts.length > 0 ? metaParts.join(' • ') : 'No additional metadata';

        return `
            <div class="tag-item" ${seekAttr}>
                <div class="tag-item-header">
                    <div class="tag-item-title">${label}</div>
                    <div class="tag-item-meta">${timeLabel}</div>
                </div>
                <div class="tag-item-meta">
                    <span class="tag-chip">${category}</span>
                    <span class="tag-chip">${source}</span>
                    ${metaText}
                </div>
                <div class="tag-item-actions">
                    <button class="action-btn delete" onclick="deleteTag(${tagId}, event)">Delete</button>
                </div>
            </div>
        `;
    }).join('');
    tagsList.innerHTML = html;
}

async function loadTags(runName) {
    if (!runName) return;
    tagsLoadError = '';
    try {
        const params = new URLSearchParams();
        if (tagFilters.label) params.set('label', tagFilters.label);
        if (tagFilters.category) params.set('category', tagFilters.category);
        if (tagFilters.source && tagFilters.source !== 'all') params.set('source', tagFilters.source);
        const query = params.toString();
        const endpoint = query
            ? `/api/runs/${runName}/tags?${query}`
            : `/api/runs/${runName}/tags`;

        const response = await fetch(endpoint);
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        const data = await response.json();
        tags = Array.isArray(data.tags) ? data.tags : [];
        renderTags();
    } catch (error) {
        console.error('Error loading tags:', error);
        tags = [];
        tagsLoadError = `Failed loading tags: ${error.message}`;
        renderTags();
    }
}

function applyTagFilters() {
    _normalizeTagFiltersFromInputs();
    if (!currentRun) {
        renderTags();
        return;
    }
    loadTags(currentRun);
}

function resetTagFilters() {
    tagFilters = {
        label: '',
        category: '',
        source: 'all'
    };
    _syncTagFilterInputs();
    if (!currentRun) {
        renderTags();
        return;
    }
    loadTags(currentRun);
}

async function addManualTagAtCurrentTime() {
    if (!currentRun) {
        showToast('Please select a run first', 'error');
        return;
    }

    const label = tagLabelInput ? String(tagLabelInput.value || '').trim() : '';
    if (!label) {
        showToast('Tag label is required', 'error');
        if (tagLabelInput) tagLabelInput.focus();
        return;
    }

    const categoryRaw = tagCategoryInput ? String(tagCategoryInput.value || '').trim() : '';
    const notesRaw = tagNotesInput ? String(tagNotesInput.value || '').trim() : '';
    const trackRaw = tagTrackInput ? String(tagTrackInput.value || '').trim() : '';
    const confidenceRaw = tagConfidenceInput ? String(tagConfidenceInput.value || '').trim() : '';
    const currentTime = Number(videoPlayer.currentTime || 0);
    const fps = Number(videoMetadata?.fps || 30);
    const frameIdx = Math.floor(currentTime * fps);

    let trackId = null;
    if (trackRaw) {
        const parsedTrack = Number.parseInt(trackRaw, 10);
        if (!Number.isFinite(parsedTrack)) {
            showToast('Track ID must be an integer', 'error');
            return;
        }
        trackId = parsedTrack;
    }

    let confidence = null;
    if (confidenceRaw) {
        const parsedConfidence = Number.parseFloat(confidenceRaw);
        if (!Number.isFinite(parsedConfidence) || parsedConfidence < 0 || parsedConfidence > 1) {
            showToast('Confidence must be between 0 and 1', 'error');
            return;
        }
        confidence = parsedConfidence;
    }

    try {
        const response = await fetch(`/api/runs/${currentRun}/tags`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                label,
                category: categoryRaw || 'general',
                start_time: currentTime,
                end_time: currentTime,
                frame_idx: frameIdx,
                track_id: trackId,
                confidence,
                source: 'manual',
                notes: notesRaw || null,
                metadata: { created_from: 'ui_tag_controls' }
            })
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        showToast('Tag added', 'success');
        if (tagNotesInput) tagNotesInput.value = '';
        if (tagTrackInput) tagTrackInput.value = '';
        if (tagConfidenceInput) tagConfidenceInput.value = '';
        await loadTags(currentRun);
    } catch (error) {
        console.error('Error creating tag:', error);
        showToast(`Failed to add tag: ${error.message}`, 'error');
    }
}

async function deleteTag(tagId, clickEvent) {
    if (clickEvent) clickEvent.stopPropagation();
    if (!currentRun) return;
    const confirmed = await showConfirmModal('Delete this tag?');
    if (!confirmed) return;

    try {
        const response = await fetch(`/api/runs/${currentRun}/tags/${tagId}`, {
            method: 'DELETE'
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        showToast('Tag deleted', 'success');
        await loadTags(currentRun);
    } catch (error) {
        console.error('Error deleting tag:', error);
        showToast(`Failed to delete tag: ${error.message}`, 'error');
    }
}

// Load score timeline
async function loadTimeline(runName) {
    try {
        const response = await fetch(`/api/runs/${runName}/timeline`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        timeline = await response.json();

        // Update score display
        if (timeline && timeline.final_score) {
            scoreDisplay.innerHTML = `
                <div class="score-display">
                    ${timeline.final_score.team_a} - ${timeline.final_score.team_b}
                </div>
            `;
        }
    } catch (error) {
        console.error('Error loading timeline:', error);
        timeline = null;
    }
}

// Load per-player reels for a run
async function loadPlayerReels(runName) {
    try {
        const response = await fetch(`/api/runs/${runName}/player_reels`);

        if (response.status === 404) {
            playerReels = [];
            playerReelsSummaryData = {};
            pruneSelectedReelPlayersToCurrentRun();
            renderPlayerReels(playerReelsSummaryData);
            return;
        }

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        playerReels = data.players || [];
        playerReelsSummaryData = data.summary || {};
        pruneSelectedReelPlayersToCurrentRun();
        renderPlayerReels(playerReelsSummaryData);
    } catch (error) {
        console.error('Error loading player reels:', error);
        playerReels = [];
        playerReelsSummaryData = {};
        pruneSelectedReelPlayersToCurrentRun();
        renderPlayerReels(playerReelsSummaryData);
    }
}

function formatPercent(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return 'n/a';
    return `${(numeric * 100).toFixed(1)}%`;
}

function setSeasonStatus(message, isError = false) {
    if (!seasonSummaryStatus) return;
    seasonSummaryStatus.textContent = message || '';
    seasonSummaryStatus.classList.toggle('status-error', isError);
    seasonSummaryStatus.classList.toggle('status-info', !isError);
}

function renderCrossMatchReport() {
    if (!seasonSummaryGrid || !seasonTeamTrends || !seasonTopPlayers || !seasonRecentWindow) {
        return;
    }

    const report = crossMatchReportData;
    if (!report || typeof report !== 'object') {
        seasonSummaryGrid.innerHTML = '<p class="loading">No season report loaded for this run</p>';
        seasonTeamTrends.innerHTML = '<p class="loading">No team trend data loaded</p>';
        seasonTopPlayers.innerHTML = '<p class="loading">No player trend data loaded</p>';
        seasonRecentWindow.innerHTML = '<p class="loading">No recent window data loaded</p>';
        setSeasonStatus('No season report loaded');
        return;
    }

    const summary = report.summary || {};
    const trends = report.season_trends || {};
    const aggregates = trends.match_aggregates || {};
    const teamTrends = trends.team_trends || {};
    const players = (report.players && Array.isArray(report.players.top_players))
        ? report.players.top_players
        : [];
    const window = trends.window || {};
    const goalsWindow = Array.isArray(window.goals) ? window.goals : [];
    const shotsWindow = Array.isArray(window.shots) ? window.shots : [];
    const highlightsWindow = Array.isArray(window.highlights) ? window.highlights : [];

    seasonSummaryGrid.innerHTML = `
        <div class="season-kpi">
            <div class="season-kpi-label">Matches Analyzed</div>
            <div class="season-kpi-value">${Number(summary.matches_analyzed || 0)}</div>
        </div>
        <div class="season-kpi">
            <div class="season-kpi-label">Unique Players</div>
            <div class="season-kpi-value">${Number(summary.unique_players || 0)}</div>
        </div>
        <div class="season-kpi">
            <div class="season-kpi-label">Goals / Match</div>
            <div class="season-kpi-value">${Number(aggregates.goals_per_match || 0).toFixed(2)}</div>
        </div>
        <div class="season-kpi">
            <div class="season-kpi-label">Shots / Match</div>
            <div class="season-kpi-value">${Number(aggregates.shots_per_match || 0).toFixed(2)}</div>
        </div>
        <div class="season-kpi">
            <div class="season-kpi-label">Highlights / Match</div>
            <div class="season-kpi-value">${Number(aggregates.highlights_per_match || 0).toFixed(2)}</div>
        </div>
        <div class="season-kpi">
            <div class="season-kpi-label">Passes / Match (Inferred)</div>
            <div class="season-kpi-value">${Number(aggregates.passes_inferred_per_match || 0).toFixed(2)}</div>
        </div>
    `;

    const teamRows = Object.entries(teamTrends);
    if (teamRows.length === 0) {
        seasonTeamTrends.innerHTML = '<p class="loading">No team trend rows available</p>';
    } else {
        seasonTeamTrends.innerHTML = teamRows.map(([team, row]) => `
            <div class="season-row">
                <strong>${escapeHtml(team)}</strong><br>
                matches ${Number(row.matches_seen || 0)} • possession ${formatPercent(row.avg_possession_share)}<br>
                high-press ${formatPercent(row.avg_high_press_rate)} • inferred passes ${Number(row.avg_passes_inferred || 0).toFixed(1)}
            </div>
        `).join('');
    }

    if (players.length === 0) {
        seasonTopPlayers.innerHTML = '<p class="loading">No top-player trend rows available</p>';
    } else {
        seasonTopPlayers.innerHTML = players.map((row, index) => `
            <div class="season-row">
                <strong>${index + 1}. ${escapeHtml(row.player_name || `Player ${row.player_id}`)}</strong><br>
                segments ${Number(row.total_segments || 0)} • matches ${Number(row.matches_with_reels || 0)} • share ${formatPercent(row.share_of_all_segments)}<br>
                avg score ${Number(row.avg_segment_score || 0).toFixed(2)} • best ${Number(row.best_segment_score || 0).toFixed(2)} • goals ${Number(row.goal_tagged_segments || 0)}
            </div>
        `).join('');
    }

    seasonRecentWindow.innerHTML = `
        <div class="season-row"><strong>Goals</strong> last ${Number(window.last_n || goalsWindow.length || 0)}: ${escapeHtml(goalsWindow.join(', ') || 'n/a')}</div>
        <div class="season-row"><strong>Shots</strong> last ${Number(window.last_n || shotsWindow.length || 0)}: ${escapeHtml(shotsWindow.join(', ') || 'n/a')}</div>
        <div class="season-row"><strong>Highlights</strong> last ${Number(window.last_n || highlightsWindow.length || 0)}: ${escapeHtml(highlightsWindow.join(', ') || 'n/a')}</div>
    `;

    const artifactCount = Object.keys(crossMatchArtifacts || {}).length;
    const generatedAt = report.generated_at ? formatIsoTimestamp(report.generated_at) : 'unknown';
    setSeasonStatus(`Season report loaded • ${artifactCount} artifacts • updated ${generatedAt}`);
}

async function loadCrossMatchReport(runName) {
    if (!runName) return;

    try {
        const response = await fetch(`/api/runs/${runName}/cross_match`);
        if (response.status === 404) {
            crossMatchReportData = null;
            crossMatchArtifacts = {};
            renderCrossMatchReport();
            setSeasonStatus('No cross-match report artifacts found for this run');
            return;
        }
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        crossMatchReportData = data.report || null;
        crossMatchArtifacts = data.available_artifacts || {};
        renderCrossMatchReport();
    } catch (error) {
        console.error('Error loading cross-match report:', error);
        crossMatchReportData = null;
        crossMatchArtifacts = {};
        renderCrossMatchReport();
        setSeasonStatus(`Failed loading season report: ${error.message}`, true);
    }
}

function _downloadFromUrl(url, fileName) {
    const link = document.createElement('a');
    link.href = url;
    if (fileName) {
        link.download = fileName;
    }
    document.body.appendChild(link);
    link.click();
    link.remove();
}

function downloadCrossMatchArtifact(artifactId) {
    if (!currentRun) return;
    const artifact = (crossMatchArtifacts && crossMatchArtifacts[artifactId]) || null;
    if (!artifact) {
        setSeasonStatus('Artifact not available for this run', true);
        return;
    }
    const url = artifact.download_url || `/api/runs/${encodeURIComponent(currentRun)}/cross_match/artifacts/${encodeURIComponent(artifactId)}`;
    _downloadFromUrl(url, artifact.file_name || '');
}

async function exportCrossMatchPackage() {
    if (!currentRun) return;

    try {
        setSeasonStatus('Building season export package...');
        const includeTemplates = crossMatchIncludeTemplates ? Boolean(crossMatchIncludeTemplates.checked) : true;
        const response = await fetch(`/api/runs/${currentRun}/cross_match/actions/export_package`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ include_templates: includeTemplates })
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        const downloadUrl = data.download_url || `/api/runs/${encodeURIComponent(currentRun)}/cross_match/exports/${encodeURIComponent(data.export_name || '')}`;
        _downloadFromUrl(downloadUrl, data.export_name || 'cross_match_export.zip');

        const summary = data.summary || {};
        setSeasonStatus(
            `Season export ready: ${Number(summary.matches_analyzed || 0)} matches, ${Number(summary.unique_players || 0)} players, ${Number(summary.artifact_files || 0)} files`
        );
    } catch (error) {
        console.error('Error exporting cross-match package:', error);
        setSeasonStatus(`Failed exporting season package: ${error.message}`, true);
    }
}

function formatPlayerName(player) {
    const jersey = player.jersey_number != null ? `#${player.jersey_number}` : null;
    const name = player.player_name || player.name || null;

    if (jersey && name) return `${jersey} ${name}`;
    if (name) return name;
    if (jersey) return jersey;
    return `Player ${player.player_id}`;
}

function formatSegmentReasons(segment) {
    const reasons = Array.isArray(segment.reasons) ? segment.reasons : [];
    if (reasons.length === 0) return 'highlight';
    return reasons.slice(0, 2).join(', ');
}

function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

function updatePlayerReelFilters() {
    if (reelTeamFilter) {
        playerReelFilters.team = reelTeamFilter.value || 'all';
    }
    if (reelSortBy) {
        playerReelFilters.sortBy = reelSortBy.value || 'best_score_desc';
    }
    if (reelMinScore) {
        const value = Number(reelMinScore.value);
        playerReelFilters.minScore = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
    }
    if (reelTopN) {
        const value = Number(reelTopN.value);
        playerReelFilters.topN = Number.isFinite(value) ? Math.max(1, Math.min(30, Math.floor(value))) : 8;
    }

    if (Array.isArray(playerReels)) {
        renderPlayerReels(playerReelsSummaryData);
    }
    updateUrlHash();
}

function getFilteredPlayerReels() {
    const teamFilter = playerReelFilters.team || 'all';
    const minScore = playerReelFilters.minScore ?? 0;
    const topN = playerReelFilters.topN ?? 8;
    const sortBy = playerReelFilters.sortBy || 'best_score_desc';

    const allPlayers = Array.isArray(playerReels) ? playerReels : [];
    const filteredPlayers = [];
    allPlayers.forEach(player => {
        const teamHint = player.team_hint === 'ours' || player.team_hint === 'opponent'
            ? player.team_hint
            : 'unknown';
        if (teamFilter !== 'all' && teamHint !== teamFilter) {
            return;
        }

        const segments = (player.segments || [])
            .filter(segment => Number(segment.player_segment_score || 0) >= minScore)
            .sort((a, b) => Number(b.player_segment_score || 0) - Number(a.player_segment_score || 0))
            .slice(0, topN);

        if (segments.length === 0) {
            return;
        }

        const maxScore = segments.reduce(
            (best, segment) => Math.max(best, Number(segment.player_segment_score || 0)),
            0
        );

        filteredPlayers.push({
            ...player,
            _displaySegments: segments,
            _teamHint: teamHint,
            _maxScore: maxScore
        });
    });

    if (sortBy === 'name_asc') {
        filteredPlayers.sort((a, b) => formatPlayerName(a).localeCompare(formatPlayerName(b)));
    } else if (sortBy === 'segment_count_desc') {
        filteredPlayers.sort((a, b) => {
            const byCount = b._displaySegments.length - a._displaySegments.length;
            if (byCount !== 0) return byCount;
            return b._maxScore - a._maxScore;
        });
    } else if (sortBy === 'player_id_asc') {
        filteredPlayers.sort((a, b) => Number(a.player_id || 0) - Number(b.player_id || 0));
    } else {
        filteredPlayers.sort((a, b) => b._maxScore - a._maxScore);
    }

    return filteredPlayers;
}

function updateSelectedReelPlayerCount() {
    if (!reelSelectedCount) return;
    reelSelectedCount.textContent = `${selectedReelPlayerIds.size} selected`;
}

function pruneSelectedReelPlayersToCurrentRun() {
    const validPlayerIds = new Set(
        (playerReels || [])
            .map(player => Number(player.player_id))
            .filter(playerId => Number.isFinite(playerId))
    );
    const nextSet = new Set();
    selectedReelPlayerIds.forEach(playerId => {
        if (validPlayerIds.has(playerId)) {
            nextSet.add(playerId);
        }
    });
    selectedReelPlayerIds = nextSet;
}

function toggleReelPlayerSelection(playerId, checked) {
    const normalizedPlayerId = Number(playerId);
    if (!Number.isFinite(normalizedPlayerId)) return;
    if (checked) {
        selectedReelPlayerIds.add(normalizedPlayerId);
    } else {
        selectedReelPlayerIds.delete(normalizedPlayerId);
    }
    updateSelectedReelPlayerCount();
}

function selectAllVisibleReelPlayers() {
    const filteredPlayers = getFilteredPlayerReels();
    filteredPlayers.forEach(player => {
        const playerId = Number(player.player_id);
        if (Number.isFinite(playerId)) {
            selectedReelPlayerIds.add(playerId);
        }
    });
    renderPlayerReels(playerReelsSummaryData);
}

function clearSelectedReelPlayers() {
    selectedReelPlayerIds.clear();
    renderPlayerReels(playerReelsSummaryData);
}

async function exportFilteredPlayerReelsPackage() {
    if (!currentRun) return;

    try {
        setIdentityStatus('Building player reel export package...');
        const selectedPlayerIds = Array.from(selectedReelPlayerIds).filter(playerId => Number.isFinite(playerId));
        const payload = {
            team_filter: playerReelFilters.team || 'all',
            min_score: Number(playerReelFilters.minScore ?? 0),
            top_n: Number(playerReelFilters.topN ?? 8),
            sort_by: playerReelFilters.sortBy || 'best_score_desc',
            include_clips: reelExportIncludeClips ? Boolean(reelExportIncludeClips.checked) : true
        };
        if (selectedPlayerIds.length > 0) {
            payload.player_ids = selectedPlayerIds;
        }
        const response = await fetch(`/api/runs/${currentRun}/player_reels/actions/export_package`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        const summary = data.summary || {};
        const players = Number(summary.players_with_reels ?? 0);
        const segments = Number(summary.player_segments_total ?? 0);
        const clips = Number(summary.clip_files_included ?? 0);
        const selectedLabel = selectedPlayerIds.length > 0 ? ` • ${selectedPlayerIds.length} selected players` : '';
        const exportName = String(data.export_name || 'player_reels_export.zip');
        setIdentityStatus(`Export ready: ${players} players, ${segments} segments, ${clips} clips${selectedLabel}`);

        const downloadUrl = data.download_url || `/api/runs/${encodeURIComponent(currentRun)}/player_reels/exports/${encodeURIComponent(exportName)}`;
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = exportName;
        document.body.appendChild(link);
        link.click();
        link.remove();
    } catch (error) {
        console.error('Error exporting player reels package:', error);
        setIdentityStatus(`Failed exporting player reels: ${error.message}`, true);
    }
}

function renderPlayerReels(summary = {}) {
    if (!playerReelsList || !playerReelsSummary) {
        return;
    }

    const allPlayers = Array.isArray(playerReels) ? playerReels : [];
    const filteredPlayers = getFilteredPlayerReels();
    updateSelectedReelPlayerCount();

    const basePlayersWithReels = summary.players_with_reels ?? allPlayers.length;
    const baseSegmentTotal = summary.player_segments_total ?? allPlayers.reduce(
        (acc, player) => acc + (player.segment_count || (player.segments || []).length || 0),
        0
    );
    const playersWithReels = filteredPlayers.length;
    const segmentTotal = filteredPlayers.reduce(
        (acc, player) => acc + player._displaySegments.length,
        0
    );

    playerReelsSummary.textContent = `${playersWithReels}/${basePlayersWithReels} players • ${segmentTotal}/${baseSegmentTotal} segments`;

    if (filteredPlayers.length === 0) {
        playerReelsList.innerHTML = '<p class="loading">No reel segments match the current filters</p>';
        return;
    }

    let html = '';
    filteredPlayers.forEach(player => {
        const playerLabel = escapeHtml(formatPlayerName(player));
        const teamHint = player._teamHint ? ` • ${escapeHtml(player._teamHint)}` : '';
        const segmentCount = player._displaySegments.length;
        const playerId = Number(player.player_id);
        const canSelect = Number.isFinite(playerId);
        const isSelected = canSelect && selectedReelPlayerIds.has(playerId);
        const selectControl = canSelect
            ? `
                <label class="player-reel-select">
                    <input
                        type="checkbox"
                        ${isSelected ? 'checked' : ''}
                        onchange="toggleReelPlayerSelection(${playerId}, this.checked)"
                    >
                    Export
                </label>
            `
            : '';

        let segmentHtml = '';
        player._displaySegments.forEach(segment => {
            const segmentPlayerId = Number(player.player_id || 0);
            const startTime = Number(segment.start_time || 0);
            const endTime = Number(segment.end_time || startTime);
            const score = Number(segment.player_segment_score || 0).toFixed(2);
            const detail = escapeHtml(formatSegmentReasons(segment));
            const encodedSegmentId = encodeURIComponent(String(segment.segment_id || ''));
            const clipAction = segment.has_clip
                ? `
                    <div class="reel-segment-actions">
                        <button
                            class="reel-clip-btn"
                            onclick="playPlayerClip(${segmentPlayerId}, '${encodedSegmentId}', event)"
                            title="Play rendered clip"
                        >
                            Play Clip
                        </button>
                    </div>
                `
                : '';

            segmentHtml += `
                <div>
                    <button
                        class="reel-segment-btn"
                        onclick="playPlayerSegment(${startTime}, ${endTime})"
                        title="Play ${formatTime(startTime)} to ${formatTime(endTime)} in main video"
                    >
                        <div class="reel-segment-time">${formatTime(startTime)} - ${formatTime(endTime)}</div>
                        <div class="reel-segment-details">score ${score} • ${detail}</div>
                    </button>
                    ${clipAction}
                </div>
            `;
        });

        html += `
            <div class="player-reel-card">
                <div class="player-reel-title">
                    <div class="player-reel-name">${playerLabel}</div>
                    <div class="player-reel-meta">${segmentCount} segments${teamHint} ${selectControl}</div>
                </div>
                ${segmentHtml}
            </div>
        `;
    });

    playerReelsList.innerHTML = html;
}

function setIdentityStatus(message, isError = false) {
    if (!identityStatus) return;
    identityStatus.textContent = message || '';
    identityStatus.classList.toggle('status-error', isError);
    identityStatus.classList.toggle('status-info', !isError);
    if (isError && message) showToast(message, 'error');
    else if (message && !message.startsWith('Loading') && !message.startsWith('Building') && !message.startsWith('Computing') && !message.startsWith('Applying') && !message.startsWith('Recomputing') && !message.startsWith('Approving')) showToast(message, 'success', 2500);
}

function renderRecomputePreview() {
    if (!recomputePreview) return;

    if (!recomputePreviewData || !recomputePreviewData.diff) {
        recomputePreview.textContent = 'No changes preview yet';
        return;
    }

    const diff = recomputePreviewData.diff;
    const current = diff.current || {};
    const preview = diff.preview || {};
    const delta = diff.delta || {};
    const playerChanges = Array.isArray(diff.player_changes) ? diff.player_changes : [];
    const topChanges = playerChanges.slice(0, 5);
    const previewId = recomputePreviewData.preview_id ? String(recomputePreviewData.preview_id) : '';

    let changesHtml = '';
    if (topChanges.length > 0) {
        changesHtml = topChanges
            .map(change => {
                const label = change.player_name || `Player ${change.player_id}`;
                const currentCount = Number(change.current_segment_count ?? 0);
                const previewCount = Number(change.preview_segment_count ?? 0);
                const deltaCount = Number(change.delta_segment_count ?? 0);
                const deltaLabel = deltaCount >= 0 ? `+${deltaCount}` : `${deltaCount}`;
                return `<div>${escapeHtml(label)}: ${currentCount} -> ${previewCount} (${deltaLabel})</div>`;
            })
            .join('');
    } else {
        changesHtml = '<div>No per-player count changes in preview.</div>';
    }

    recomputePreview.innerHTML = `
        <strong>Recompute Preview</strong><br>
        Players with reels: ${Number(current.players_with_reels ?? 0)} -> ${Number(preview.players_with_reels ?? 0)}<br>
        Segments total: ${Number(current.player_segments_total ?? 0)} -> ${Number(preview.player_segments_total ?? 0)}<br>
        Delta: players ${Number(delta.players_with_reels ?? 0)}, segments ${Number(delta.player_segments_total ?? 0)}, gained ${Number(delta.gained_segments_total ?? 0)}, lost ${Number(delta.lost_segments_total ?? 0)}
        ${previewId ? `<div style="margin-top: 0.4rem;"><strong>Preview ID</strong> ${escapeHtml(previewId)}</div>` : ''}
        <div style="margin-top: 0.4rem;"><strong>Top Changes</strong></div>
        ${changesHtml}
    `;
}

function clearRecomputePreview() {
    recomputePreviewData = null;
    renderRecomputePreview();
}

function formatIsoTimestamp(value) {
    if (!value) return 'unknown time';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString();
}

function summarizeEditAction(edit) {
    const action = edit.action || 'unknown';
    if (action === 'assign') return 'Assign';
    if (action === 'bulk_assign') return 'Bulk Assign';
    if (action === 'undo') return 'Undo';
    return action;
}

function summarizeEditCounts(edit) {
    const summary = edit.summary || {};
    if (edit.action === 'undo') {
        const reverted = Number(summary.reverted_count ?? 0);
        const deleted = Number(summary.deleted_count ?? 0);
        const failed = Number(summary.failed_count ?? 0);
        return `reverted ${reverted}, deleted ${deleted}, failed ${failed}`;
    }
    const updated = Number(summary.updated_count ?? 0);
    const created = Number(summary.created_count ?? 0);
    const missing = Number(summary.missing_count ?? 0);
    const failed = Number(summary.failed_count ?? 0);
    return `updated ${updated}, created ${created}, missing ${missing}, failed ${failed}`;
}

function renderIdentityEdits() {
    if (!identityEditsList) return;

    if (!Array.isArray(identityEditsData) || identityEditsData.length === 0) {
        identityEditsList.innerHTML = '<p class="loading">No identity edits yet</p>';
        return;
    }

    let html = '';
    identityEditsData.forEach(edit => {
        const opId = String(edit.op_id || '');
        const action = summarizeEditAction(edit);
        const trackCount = Number(edit.track_count ?? (Array.isArray(edit.track_ids) ? edit.track_ids.length : 0));
        const playerLabel = edit.player_id == null ? 'None' : `Player ${edit.player_id}`;
        const recordedAt = formatIsoTimestamp(edit.recorded_at);
        const countsLabel = summarizeEditCounts(edit);
        const targetLabel = edit.target_op_id ? `target ${escapeHtml(edit.target_op_id)}` : '';
        const undoButton = (edit.undoable && opId)
            ? `<button class="identity-btn" onclick="undoIdentityEditByOperation('${encodeURIComponent(opId)}')">Undo</button>`
            : '';

        html += `
            <div class="identity-edit-row">
                <div class="identity-edit-meta">
                    <strong>${escapeHtml(action)}</strong> • ${trackCount} tracks • ${escapeHtml(playerLabel)}<br>
                    ${escapeHtml(countsLabel)}<br>
                    ${escapeHtml(recordedAt)} ${targetLabel}
                </div>
                ${undoButton}
            </div>
        `;
    });

    identityEditsList.innerHTML = html;
}

function updateSelectedSuggestionCount() {
    if (!suggestionsSelectedCount) return;
    suggestionsSelectedCount.textContent = `${selectedSuggestionTrackIds.size} selected`;
}

function pruneSelectedSuggestionsToCurrentRun() {
    const validTrackIds = new Set(
        (identitySuggestionsData || [])
            .map(row => Number(row.track_id))
            .filter(trackId => Number.isFinite(trackId))
    );
    const nextSet = new Set();
    selectedSuggestionTrackIds.forEach(trackId => {
        if (validTrackIds.has(trackId)) {
            nextSet.add(trackId);
        }
    });
    selectedSuggestionTrackIds = nextSet;
}

function formatSuggestionCandidate(candidate) {
    const name = candidate.player_name || `Player ${candidate.player_id}`;
    const score = Number(candidate.score ?? 0).toFixed(2);
    const body = Number(candidate.reason_breakdown?.body_reid ?? 0).toFixed(2);
    const profile = Number(candidate.reason_breakdown?.profile_match ?? 0).toFixed(2);
    const bonus = Number(candidate.reason_breakdown?.agreement_bonus ?? 0).toFixed(2);
    return escapeHtml(`${name} (${score}) b:${body} p:${profile} g:${bonus}`);
}

function getTrackThumbnailUrl(trackId) {
    if (!currentRun) return '';
    const encodedRun = encodeURIComponent(currentRun);
    const encodedTrackId = encodeURIComponent(String(trackId));
    return `/api/runs/${encodedRun}/tracks/${encodedTrackId}/thumbnail`;
}

function jumpToTrackFrame(frameStart, fps = 25) {
    const frame = Number(frameStart);
    const frameRate = Number(fps);
    if (!Number.isFinite(frame) || !Number.isFinite(frameRate) || frameRate <= 0) {
        return;
    }

    const targetSeconds = Math.max(0, frame / frameRate);
    showView('matchAnalysisView');
    activeSegmentEndTime = null;

    if (typeof seekVideoTo === 'function') {
        seekVideoTo(targetSeconds);
    } else if (typeof _seekAndPlay === 'function') {
        _seekAndPlay(targetSeconds);
    } else if (videoPlayer) {
        videoPlayer.currentTime = targetSeconds;
        videoPlayer.play().catch(() => {});
    }
    updateUrlHash(targetSeconds);

    if (viewer && typeof viewer.scrollIntoView === 'function') {
        requestAnimationFrame(() => {
            viewer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    }
}

function renderIdentitySuggestions() {
    if (!identitySuggestionsList) return;

    if (!Array.isArray(identitySuggestionsData) || identitySuggestionsData.length === 0) {
        identitySuggestionsList.innerHTML = '<p class="loading">No suggestions available for this run</p>';
        updateSelectedSuggestionCount();
        return;
    }

    const sugPageEnd = (suggestionPage + 1) * SUGGESTION_PAGE_SIZE;
    const visibleSuggestions = identitySuggestionsData.slice(0, sugPageEnd);
    const sugRemaining = identitySuggestionsData.length - visibleSuggestions.length;

    let html = '';
    visibleSuggestions.forEach(row => {
        const trackId = Number(row.track_id);
        const isSelected = selectedSuggestionTrackIds.has(trackId);
        const frameStart = Number(row.frame_start);
        const hasFrameStart = Number.isFinite(frameStart);
        const recommended = row.recommended || {};
        const playerName = recommended.player_name || (recommended.player_id != null ? `Player ${recommended.player_id}` : 'None');
        const confidence = Number(recommended.confidence ?? 0).toFixed(2);
        const method = recommended.method || 'unknown';
        const status = row.status || 'pending';
        const needsReview = Boolean(row.needs_review);
        const candidates = Array.isArray(row.candidates) ? row.candidates.slice(0, 2) : [];
        const candidateText = candidates.length > 0
            ? candidates.map(formatSuggestionCandidate).join('<br>')
            : 'No candidates';
        const disabledCheckbox = (!needsReview || status === 'applied') ? 'disabled' : '';
        const checked = (isSelected && !disabledCheckbox) ? 'checked' : '';
        const thumbUrl = getTrackThumbnailUrl(trackId);
        const jumpDisabled = hasFrameStart ? '' : 'disabled title="No frame start available"';

        html += `
            <div class="identity-suggestion-row">
                <div class="identity-select-cell">
                    <input
                        type="checkbox"
                        ${checked}
                        ${disabledCheckbox}
                        onchange="toggleSuggestionSelection(${trackId}, this.checked)"
                    >
                </div>
                <div class="identity-row-main">
                    <img class="track-thumb" src="${thumbUrl}" loading="lazy" alt="Track ${trackId}">
                    <div class="identity-row-text">
                        <div class="identity-id">Track ${trackId}</div>
                        <div class="identity-suggestion-meta">
                            ${escapeHtml(playerName)} • conf ${escapeHtml(confidence)} • ${escapeHtml(method)}<br>
                            status ${escapeHtml(status)} • strategy ${escapeHtml(row.fusion_strategy || 'body_only')}
                        </div>
                        <div class="identity-suggestion-meta">${candidateText}</div>
                    </div>
                </div>
                <button class="identity-btn identity-jump-btn" ${jumpDisabled} onclick="jumpToTrackFrame(${hasFrameStart ? frameStart : 'null'}, 25)">▶</button>
            </div>
        `;
    });

    if (sugRemaining > 0) {
        html += `<button class="show-more-btn" onclick="suggestionPage++;renderIdentitySuggestions()">Show more (${sugRemaining} remaining)</button>`;
    }

    identitySuggestionsList.innerHTML = html;
    updateSelectedSuggestionCount();
}

function identityPlayerLabel(player) {
    return formatPlayerName(player);
}

function identityPlayerOptions(selectedPlayerId = null) {
    const players = identityReviewData.players || [];
    let html = '<option value="">Unassigned</option>';
    players.forEach(player => {
        const playerId = Number(player.player_id);
        const selected = selectedPlayerId != null && Number(selectedPlayerId) === playerId ? 'selected' : '';
        html += `<option value="${playerId}" ${selected}>${escapeHtml(identityPlayerLabel(player))}</option>`;
    });
    return html;
}

function identityPlayerOptionsForBulk() {
    const players = identityReviewData.players || [];
    let html = '<option value="">Assign selected to...</option>';
    html += '<option value="__unassign__">Unassign selected</option>';
    players.forEach(player => {
        const playerId = Number(player.player_id);
        html += `<option value="${playerId}">${escapeHtml(identityPlayerLabel(player))}</option>`;
    });
    return html;
}

function updateSelectedAssignmentCount() {
    if (!bulkSelectedCount) return;
    bulkSelectedCount.textContent = `${selectedAssignmentTrackIds.size} selected`;
}

function pruneSelectedAssignmentsToCurrentRun() {
    const validTrackIds = new Set(
        (identityReviewData.assignments || [])
            .map(row => Number(row.track_id))
            .filter(trackId => Number.isFinite(trackId))
    );
    const nextSet = new Set();
    selectedAssignmentTrackIds.forEach(trackId => {
        if (validTrackIds.has(trackId)) {
            nextSet.add(trackId);
        }
    });
    selectedAssignmentTrackIds = nextSet;
}

function getFilteredAssignments() {
    const assignments = identityReviewData.assignments || [];
    const filterTerm = (assignmentFilterTerm || '').trim().toLowerCase();

    return assignments.filter(row => {
        if (!filterTerm) return true;
        const trackId = String(row.track_id ?? '');
        const playerName = String(row.player_name ?? '').toLowerCase();
        const playerId = String(row.player_id ?? '');
        return trackId.includes(filterTerm) || playerName.includes(filterTerm) || playerId.includes(filterTerm);
    });
}

function renderIdentityMergeOptions() {
    if (!mergeKeepPlayer || !mergeFromPlayer) return;

    const players = identityReviewData.players || [];
    let options = '<option value="">Select player</option>';
    players.forEach(player => {
        const playerId = Number(player.player_id);
        options += `<option value="${playerId}">${escapeHtml(identityPlayerLabel(player))}</option>`;
    });

    mergeKeepPlayer.innerHTML = options;
    mergeFromPlayer.innerHTML = options;

    if (bulkAssignPlayer) {
        const selectedValue = bulkAssignPlayer.value;
        bulkAssignPlayer.innerHTML = identityPlayerOptionsForBulk();
        if (selectedValue) {
            bulkAssignPlayer.value = selectedValue;
        }
    }
}

function renderIdentityPlayers() {
    if (!identityPlayersList) return;

    const players = identityReviewData.players || [];
    if (players.length === 0) {
        identityPlayersList.innerHTML = '<p class="loading">No players in identity database yet</p>';
        return;
    }

    let html = '';
    players.forEach(player => {
        const playerId = Number(player.player_id);
        const nameValue = escapeHtml(player.name || '');
        const jerseyValue = player.jersey_number != null ? Number(player.jersey_number) : '';
        const teamHint = player.team_hint || '';

        html += `
            <div class="identity-player-row">
                <div class="identity-id">ID ${playerId}</div>
                <input id="playerName_${playerId}" class="form-input" type="text" value="${nameValue}" placeholder="Name">
                <input id="playerJersey_${playerId}" class="form-input" type="number" value="${jerseyValue}" placeholder="Jersey">
                <select id="playerTeam_${playerId}" class="form-input">
                    <option value="" ${teamHint === '' ? 'selected' : ''}>Unknown</option>
                    <option value="ours" ${teamHint === 'ours' ? 'selected' : ''}>Ours</option>
                    <option value="opponent" ${teamHint === 'opponent' ? 'selected' : ''}>Opponent</option>
                </select>
                <button class="identity-btn" onclick="savePlayerMetadata(${playerId})">Save</button>
            </div>
        `;
    });

    identityPlayersList.innerHTML = html;
}

function renderIdentityAssignments() {
    if (!identityAssignmentsList) return;

    const filtered = getFilteredAssignments();

    if (filtered.length === 0) {
        identityAssignmentsList.innerHTML = '<p class="loading">No assignments match filter</p>';
        updateSelectedAssignmentCount();
        return;
    }

    const pageEnd = (assignmentPage + 1) * ASSIGNMENT_PAGE_SIZE;
    const visible = filtered.slice(0, pageEnd);
    const remaining = filtered.length - visible.length;

    let html = '';
    visible.forEach(row => {
        const trackId = Number(row.track_id);
        const isSelected = selectedAssignmentTrackIds.has(trackId);
        const currentPlayerId = row.player_id != null ? Number(row.player_id) : null;
        const matchMethod = row.match_method || 'unknown';
        const matchConfidence = row.match_confidence != null ? Number(row.match_confidence).toFixed(2) : 'n/a';
        const frameStart = row.frame_start != null ? row.frame_start : '?';
        const frameEnd = row.frame_end != null ? row.frame_end : '?';
        const lockState = row.lock_state || 'candidate';
        const lockReason = row.lock_reason ? String(row.lock_reason) : '';
        const lockConflictTrack = row.lock_conflict_with_track_id != null
            ? Number(row.lock_conflict_with_track_id)
            : null;
        const fusionStrategy = row.fusion_strategy || 'body_only';
        const frameStartNum = Number(frameStart);
        const hasFrameStart = Number.isFinite(frameStartNum);
        const thumbUrl = getTrackThumbnailUrl(trackId);

        const faceConf = row.face_confidence != null ? Number(row.face_confidence).toFixed(2) : null;
        const faceSupport = row.face_support_frames != null ? Number(row.face_support_frames) : null;
        const faceBackend = row.face_backend ? String(row.face_backend) : null;
        const jerseyNumber = row.jersey_number_detected != null ? Number(row.jersey_number_detected) : null;
        const jerseyConf = row.jersey_ocr_confidence != null ? Number(row.jersey_ocr_confidence).toFixed(2) : null;
        const jerseySupport = row.jersey_ocr_support_frames != null ? Number(row.jersey_ocr_support_frames) : null;
        const jerseyAmbiguous = row.jersey_ocr_ambiguous === true;
        const appliedSignals = Array.isArray(row.multimodal_applied) ? row.multimodal_applied : [];

        const lockParts = [`lock ${lockState}`];
        if (lockReason) lockParts.push(lockReason);
        if (lockConflictTrack != null) lockParts.push(`conflicts track ${lockConflictTrack}`);
        const lockText = lockParts.join(' • ');

        const evidenceParts = [];
        if (faceConf != null) {
            const faceMeta = [
                `face ${faceConf}`,
                faceSupport != null ? `n=${faceSupport}` : null,
                faceBackend
            ].filter(Boolean).join(' ');
            evidenceParts.push(faceMeta);
        }
        if (jerseyNumber != null || jerseyConf != null) {
            const jerseyMeta = [
                jerseyNumber != null ? `jersey #${jerseyNumber}` : 'jersey',
                jerseyConf != null ? jerseyConf : null,
                jerseySupport != null ? `n=${jerseySupport}` : null,
                jerseyAmbiguous ? 'ambiguous' : null
            ].filter(Boolean).join(' ');
            evidenceParts.push(jerseyMeta);
        }
        if (appliedSignals.length > 0) {
            evidenceParts.push(`applied ${appliedSignals.join(',')}`);
        }
        if (evidenceParts.length === 0) {
            evidenceParts.push('no extra multimodal evidence');
        }
        const evidenceText = evidenceParts.join(' • ');
        const jumpDisabled = hasFrameStart ? '' : 'disabled title="No frame start available"';

        html += `
            <div class="identity-assignment-row">
                <div class="identity-select-cell">
                    <input
                        type="checkbox"
                        ${isSelected ? 'checked' : ''}
                        onchange="toggleAssignmentSelection(${trackId}, this.checked)"
                    >
                </div>
                <div class="identity-row-main">
                    <img class="track-thumb" src="${thumbUrl}" loading="lazy" alt="Track ${trackId}">
                    <div class="identity-row-text">
                        <div class="identity-id">Track ${trackId}</div>
                        <div class="identity-assignment-meta">
                            ${escapeHtml(matchMethod)} • conf ${escapeHtml(matchConfidence)}<br>
                            frames ${escapeHtml(frameStart)}-${escapeHtml(frameEnd)}<br>
                            ${escapeHtml(lockText)}
                        </div>
                        <div class="identity-assignment-meta">
                            ${escapeHtml(row.player_name || 'Unassigned')}<br>
                            strategy ${escapeHtml(fusionStrategy)}<br>
                            ${escapeHtml(evidenceText)}
                        </div>
                    </div>
                </div>
                <select id="assignmentPlayer_${trackId}" class="form-input">
                    ${identityPlayerOptions(currentPlayerId)}
                </select>
                <button class="identity-btn" onclick="saveTrackAssignment(${trackId})">Assign</button>
                <button class="identity-btn identity-jump-btn" ${jumpDisabled} onclick="jumpToTrackFrame(${hasFrameStart ? frameStartNum : 'null'}, 25)">▶</button>
            </div>
        `;
    });

    if (remaining > 0) {
        html += `<button class="show-more-btn" onclick="assignmentPage++;renderIdentityAssignments()">Show more (${remaining} remaining)</button>`;
    }

    identityAssignmentsList.innerHTML = html;
    updateSelectedAssignmentCount();
}

function renderIdentityReview() {
    if (!identitySummary) return;

    const summary = identityReviewData.summary || {};
    const total = summary.total_assignments ?? 0;
    const assigned = summary.assigned ?? 0;
    const manual = summary.manual ?? 0;
    const locked = summary.locked ?? 0;
    const unlocked = summary.unlocked ?? 0;
    const pct = total > 0 ? Math.round((assigned / total) * 100) : 0;
    identitySummary.textContent = `${assigned} of ${total.toLocaleString()} tracks identified (${pct}%) • ${manual} manual • ${locked} locked`;

    renderIdentityMergeOptions();
    renderIdentityPlayers();
    renderIdentityAssignments();
    updateSelectedAssignmentCount();
}

async function loadIdentityReview(runName) {
    if (!runName) return;

    try {
        const response = await fetch(`/api/runs/${runName}/identity_review`);
        if (response.status === 404) {
            identityReviewData = { video_id: null, players: [], assignments: [], summary: {} };
            renderIdentityReview();
            setIdentityStatus('No identity review data available for this run');
            return;
        }
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        identityReviewData = {
            video_id: data.video_id || null,
            players: data.players || [],
            assignments: data.assignments || [],
            summary: data.summary || {}
        };
        pruneSelectedAssignmentsToCurrentRun();

        if (assignmentSearch) {
            assignmentSearch.value = '';
        }
        assignmentFilterTerm = '';
        renderIdentityReview();
        setIdentityStatus(`Loaded ${identityReviewData.assignments.length} assignments`);
    } catch (error) {
        console.error('Error loading identity review:', error);
        identityReviewData = { video_id: null, players: [], assignments: [], summary: {} };
        renderIdentityReview();
        setIdentityStatus('Failed to load identity review', true);
    }
}

async function loadIdentitySuggestions(runName) {
    if (!runName) return;

    try {
        const response = await fetch(`/api/runs/${runName}/identity_suggestions`);
        if (response.status === 404) {
            identitySuggestionsData = [];
            selectedSuggestionTrackIds.clear();
            renderIdentitySuggestions();
            return;
        }
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        identitySuggestionsData = Array.isArray(data.suggestions) ? data.suggestions : [];
        pruneSelectedSuggestionsToCurrentRun();
        renderIdentitySuggestions();
    } catch (error) {
        console.error('Error loading identity suggestions:', error);
        identitySuggestionsData = [];
        selectedSuggestionTrackIds.clear();
        renderIdentitySuggestions();
    }
}

async function loadIdentityEdits(runName) {
    if (!runName) return;

    try {
        const response = await fetch(`/api/runs/${runName}/identity_review/edits?limit=50`);
        if (response.status === 404) {
            identityEditsData = [];
            renderIdentityEdits();
            return;
        }
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        identityEditsData = Array.isArray(data.edits) ? data.edits : [];
        renderIdentityEdits();
    } catch (error) {
        console.error('Error loading identity edits:', error);
        identityEditsData = [];
        renderIdentityEdits();
    }
}

function updateAssignmentFilter() {
    assignmentFilterTerm = assignmentSearch ? assignmentSearch.value : '';
    assignmentPage = 0;
    renderIdentityAssignments();
}

function toggleAssignmentSelection(trackId, checked) {
    if (checked) {
        selectedAssignmentTrackIds.add(trackId);
    } else {
        selectedAssignmentTrackIds.delete(trackId);
    }
    updateSelectedAssignmentCount();
}

function selectAllFilteredAssignments() {
    const filtered = getFilteredAssignments();
    filtered.forEach(row => {
        const trackId = Number(row.track_id);
        if (Number.isFinite(trackId)) {
            selectedAssignmentTrackIds.add(trackId);
        }
    });
    renderIdentityAssignments();
}

function clearSelectedAssignments() {
    selectedAssignmentTrackIds.clear();
    renderIdentityAssignments();
}

function toggleSuggestionSelection(trackId, checked) {
    if (checked) {
        selectedSuggestionTrackIds.add(trackId);
    } else {
        selectedSuggestionTrackIds.delete(trackId);
    }
    updateSelectedSuggestionCount();
}

function selectAllPendingSuggestions() {
    (identitySuggestionsData || []).forEach(row => {
        const trackId = Number(row.track_id);
        if (!Number.isFinite(trackId)) return;
        if (!row.needs_review) return;
        if (String(row.status || '') === 'applied') return;
        selectedSuggestionTrackIds.add(trackId);
    });
    renderIdentitySuggestions();
}

function clearSelectedSuggestions() {
    selectedSuggestionTrackIds.clear();
    renderIdentitySuggestions();
}

function getSuggestionApplyPayload() {
    const minConfidenceRaw = suggestionMinConfidence ? Number(suggestionMinConfidence.value) : 0.7;
    const minConfidence = Number.isFinite(minConfidenceRaw)
        ? Math.max(0, Math.min(1, minConfidenceRaw))
        : 0.7;

    return {
        track_ids: Array.from(selectedSuggestionTrackIds),
        min_confidence: minConfidence,
        suggested_only: true
    };
}

function preserveExistingClipsEnabled() {
    if (!preserveExistingClipsToggle) return true;
    return Boolean(preserveExistingClipsToggle.checked);
}

async function applySelectedSuggestions() {
    if (!currentRun) return;

    if (selectedSuggestionTrackIds.size === 0) {
        setIdentityStatus('Select at least one suggestion to apply', true);
        return;
    }

    try {
        const payload = getSuggestionApplyPayload();
        const response = await fetch(`/api/runs/${currentRun}/identity_suggestions/actions/apply`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        const applied = Number(data.applied_count ?? 0);
        const skipped = Number(data.skipped_count ?? 0);
        const failed = Number(data.failed_count ?? 0);
        setIdentityStatus(`Applied suggestions: ${applied}, skipped ${skipped}, failed ${failed}`);

        selectedSuggestionTrackIds.clear();
        clearRecomputePreview();
        await loadIdentityReview(currentRun);
        await loadIdentitySuggestions(currentRun);
        await loadIdentityEdits(currentRun);
    } catch (error) {
        console.error('Error applying identity suggestions:', error);
        setIdentityStatus(`Failed applying suggestions: ${error.message}`, true);
    }
}

async function applySuggestionsAndRecompute() {
    if (!currentRun) return;

    if (selectedSuggestionTrackIds.size === 0) {
        setIdentityStatus('Select at least one suggestion to apply', true);
        return;
    }

    try {
        setIdentityStatus('Applying suggestions and recomputing reels...');
        const payload = {
            ...getSuggestionApplyPayload(),
            preserve_existing_clips: preserveExistingClipsEnabled()
        };
        const response = await fetch(`/api/runs/${currentRun}/identity_suggestions/actions/apply_and_recompute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        const apply = data.apply || {};
        const recompute = data.recompute || {};
        const applied = Number(apply.applied_count ?? 0);
        const skipped = Number(apply.skipped_count ?? 0);
        const failed = Number(apply.failed_count ?? 0);
        const players = Number(recompute.summary?.players_with_reels ?? 0);
        const segments = Number(recompute.summary?.player_segments_total ?? 0);
        setIdentityStatus(
            `Applied suggestions: ${applied}, skipped ${skipped}, failed ${failed}. Recomputed reels: ${players} players, ${segments} segments`
        );

        selectedSuggestionTrackIds.clear();
        clearRecomputePreview();
        await loadIdentityReview(currentRun);
        await loadIdentitySuggestions(currentRun);
        await loadIdentityEdits(currentRun);
        await loadPlayerReels(currentRun);
        await loadRuns();
    } catch (error) {
        console.error('Error applying suggestions and recomputing reels:', error);
        setIdentityStatus(`Failed apply + recompute: ${error.message}`, true);
    }
}

async function applySuggestionsAndPreview() {
    if (!currentRun) return;

    if (selectedSuggestionTrackIds.size === 0) {
        setIdentityStatus('Select at least one suggestion to apply', true);
        return;
    }

    try {
        setIdentityStatus('Applying suggestions and computing preview...');
        const payload = {
            ...getSuggestionApplyPayload(),
            preserve_existing_clips: preserveExistingClipsEnabled()
        };
        const response = await fetch(`/api/runs/${currentRun}/identity_suggestions/actions/apply_and_preview`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        const apply = data.apply || {};
        const preview = data.preview || {};
        recomputePreviewData = preview;
        renderRecomputePreview();

        const applied = Number(apply.applied_count ?? 0);
        const skipped = Number(apply.skipped_count ?? 0);
        const failed = Number(apply.failed_count ?? 0);
        const delta = preview.diff?.delta || {};
        setIdentityStatus(
            `Applied suggestions: ${applied}, skipped ${skipped}, failed ${failed}. Preview delta: players ${Number(delta.players_with_reels ?? 0)}, segments ${Number(delta.player_segments_total ?? 0)}`
        );

        selectedSuggestionTrackIds.clear();
        await loadIdentityReview(currentRun);
        await loadIdentitySuggestions(currentRun);
        await loadIdentityEdits(currentRun);
    } catch (error) {
        console.error('Error applying suggestions and previewing recompute diff:', error);
        setIdentityStatus(`Failed apply + preview: ${error.message}`, true);
    }
}

async function approvePreviewAndPersist() {
    if (!currentRun) return;

    const previewId = recomputePreviewData && recomputePreviewData.preview_id
        ? String(recomputePreviewData.preview_id)
        : '';
    if (!previewId) {
        setIdentityStatus('Generate a preview first, then approve it to persist', true);
        return;
    }

    try {
        setIdentityStatus('Approving preview and persisting reels...');
        const response = await fetch(`/api/runs/${currentRun}/player_reels/actions/approve_preview`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ preview_id: previewId })
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        const players = Number(data.summary?.players_with_reels ?? 0);
        const segments = Number(data.summary?.player_segments_total ?? 0);
        setIdentityStatus(`Approved preview ${previewId}. Persisted reels: ${players} players, ${segments} segments`);

        clearRecomputePreview();
        await loadPlayerReels(currentRun);
        await loadIdentityReview(currentRun);
        await loadIdentitySuggestions(currentRun);
        await loadIdentityEdits(currentRun);
        await loadRuns();
    } catch (error) {
        console.error('Error approving preview and persisting reels:', error);
        setIdentityStatus(`Failed approve preview: ${error.message}`, true);
    }
}

async function applyBulkAssignment() {
    if (!currentRun) return;

    if (selectedAssignmentTrackIds.size === 0) {
        setIdentityStatus('Select at least one track for bulk assignment', true);
        return;
    }

    const rawSelection = bulkAssignPlayer ? bulkAssignPlayer.value : '';
    if (!rawSelection) {
        setIdentityStatus('Select a target action for bulk assignment', true);
        return;
    }
    const isUnassign = rawSelection === '__unassign__';
    const playerId = isUnassign ? null : Number(rawSelection);
    if (!isUnassign && !Number.isFinite(playerId)) {
        setIdentityStatus('Select a valid target player for bulk assignment', true);
        return;
    }

    try {
        const response = await fetch(`/api/runs/${currentRun}/identity_review/actions/bulk_assign`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                track_ids: Array.from(selectedAssignmentTrackIds),
                player_id: playerId,
                confidence: 1.0,
                method: 'manual'
            })
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        const updated = data.updated_count ?? 0;
        const created = data.created_count ?? 0;
        const failed = (data.failed_count ?? 0) + (data.missing_count ?? 0);
        const operationId = data.operation_id ? ` • op ${data.operation_id}` : '';
        const actionLabel = isUnassign ? 'Bulk unassign complete' : 'Bulk assignment complete';
        setIdentityStatus(`${actionLabel}: updated ${updated}, created ${created}, unresolved ${failed}${operationId}`);

        selectedAssignmentTrackIds.clear();
        clearRecomputePreview();
        await loadIdentityReview(currentRun);
        await loadIdentitySuggestions(currentRun);
        await loadIdentityEdits(currentRun);
    } catch (error) {
        console.error('Error applying bulk assignment:', error);
        setIdentityStatus(`Bulk assignment failed: ${error.message}`, true);
    }
}

async function refreshIdentityReview() {
    if (!currentRun) return;
    await loadIdentityReview(currentRun);
    await loadIdentitySuggestions(currentRun);
    await loadIdentityEdits(currentRun);
}

async function savePlayerMetadata(playerId) {
    const nameInput = document.getElementById(`playerName_${playerId}`);
    const jerseyInput = document.getElementById(`playerJersey_${playerId}`);
    const teamInput = document.getElementById(`playerTeam_${playerId}`);

    if (!nameInput || !jerseyInput || !teamInput) return;

    const payload = {
        name: nameInput.value.trim() || null,
        jersey_number: jerseyInput.value !== '' ? Number(jerseyInput.value) : null,
        team_hint: teamInput.value || null
    };

    try {
        const response = await fetch(`/api/players/${playerId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }
        setIdentityStatus(`Updated player ${playerId}`);
        await loadIdentityReview(currentRun);
        await loadIdentitySuggestions(currentRun);
        await loadPlayerReels(currentRun);
    } catch (error) {
        console.error('Error updating player metadata:', error);
        setIdentityStatus(`Failed to update player ${playerId}: ${error.message}`, true);
    }
}

async function createPlayerFromForm() {
    const payload = {
        name: newPlayerName && newPlayerName.value.trim() ? newPlayerName.value.trim() : null,
        jersey_number: newPlayerJersey && newPlayerJersey.value !== '' ? Number(newPlayerJersey.value) : null,
        team_hint: newPlayerTeam && newPlayerTeam.value ? newPlayerTeam.value : null
    };

    try {
        const response = await fetch('/api/players', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        if (newPlayerName) newPlayerName.value = '';
        if (newPlayerJersey) newPlayerJersey.value = '';
        if (newPlayerTeam) newPlayerTeam.value = '';

        setIdentityStatus('Created new player. Reassign tracks as needed.');
        await loadIdentityReview(currentRun);
        await loadIdentitySuggestions(currentRun);
    } catch (error) {
        console.error('Error creating player:', error);
        setIdentityStatus(`Failed to create player: ${error.message}`, true);
    }
}

async function mergePlayersFromForm() {
    const keepId = mergeKeepPlayer ? Number(mergeKeepPlayer.value) : NaN;
    const mergeId = mergeFromPlayer ? Number(mergeFromPlayer.value) : NaN;

    if (!Number.isFinite(keepId) || !Number.isFinite(mergeId)) {
        setIdentityStatus('Select both players to merge', true);
        return;
    }
    if (keepId === mergeId) {
        setIdentityStatus('Keep and merge player must be different', true);
        return;
    }

    try {
        const response = await fetch(`/api/players/merge/${keepId}/${mergeId}`, {
            method: 'POST'
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        setIdentityStatus(`Merged player ${mergeId} into ${keepId}`);
        await loadIdentityReview(currentRun);
        await loadIdentitySuggestions(currentRun);
        await loadPlayerReels(currentRun);
    } catch (error) {
        console.error('Error merging players:', error);
        setIdentityStatus(`Failed to merge players: ${error.message}`, true);
    }
}

async function saveTrackAssignment(trackId) {
    if (!currentRun) return;

    const select = document.getElementById(`assignmentPlayer_${trackId}`);
    if (!select) {
        setIdentityStatus('Assignment row not found', true);
        return;
    }

    const playerId = select.value ? Number(select.value) : null;
    if (select.value && !Number.isFinite(playerId)) {
        setIdentityStatus('Invalid player selection', true);
        return;
    }

    try {
        const response = await fetch(`/api/runs/${currentRun}/identity_review/actions/assign`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                track_id: Number(trackId),
                player_id: playerId,
                confidence: 1.0,
                method: 'manual'
            })
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        const actionLabel = playerId == null ? 'Unassigned' : 'Assigned';
        setIdentityStatus(`${actionLabel} track ${trackId}${playerId == null ? '' : ` to player ${playerId}`}`);
        clearRecomputePreview();
        await loadIdentityReview(currentRun);
        await loadIdentitySuggestions(currentRun);
        await loadIdentityEdits(currentRun);
    } catch (error) {
        console.error('Error assigning track:', error);
        setIdentityStatus(`Failed assignment for track ${trackId}: ${error.message}`, true);
    }
}

async function recomputePlayerReelsFromIdentity() {
    if (!currentRun) return;

    try {
        setIdentityStatus('Recomputing player reels...');
        const response = await fetch(`/api/runs/${currentRun}/player_reels/actions/recompute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ preserve_existing_clips: preserveExistingClipsEnabled() })
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        const players = data.summary?.players_with_reels ?? 0;
        const segments = data.summary?.player_segments_total ?? 0;
        setIdentityStatus(`Recomputed reels: ${players} players, ${segments} segments`);

        clearRecomputePreview();
        await loadPlayerReels(currentRun);
        await loadIdentityReview(currentRun);
        await loadRuns();
    } catch (error) {
        console.error('Error recomputing player reels:', error);
        setIdentityStatus(`Failed to recompute reels: ${error.message}`, true);
    }
}

async function previewRecomputeDiff() {
    if (!currentRun) return;

    try {
        setIdentityStatus('Computing recompute preview...');
        const response = await fetch(`/api/runs/${currentRun}/player_reels/actions/recompute_preview`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ preserve_existing_clips: preserveExistingClipsEnabled() })
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        recomputePreviewData = data;
        renderRecomputePreview();

        const delta = data.diff?.delta || {};
        setIdentityStatus(
            `Preview ready: delta players ${Number(delta.players_with_reels ?? 0)}, delta segments ${Number(delta.player_segments_total ?? 0)}`
        );
    } catch (error) {
        console.error('Error computing recompute preview:', error);
        setIdentityStatus(`Failed preview recompute diff: ${error.message}`, true);
    }
}

async function undoLastIdentityEdit() {
    if (!currentRun) return;

    try {
        const response = await fetch(`/api/runs/${currentRun}/identity_review/actions/undo`, {
            method: 'POST'
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        const reverted = Number(data.reverted_count ?? 0);
        const deleted = Number(data.deleted_count ?? 0);
        const failed = Number(data.failed_count ?? 0);
        setIdentityStatus(`Undo complete: reverted ${reverted}, deleted ${deleted}, failed ${failed}`);

        selectedAssignmentTrackIds.clear();
        clearRecomputePreview();
        await loadIdentityReview(currentRun);
        await loadIdentitySuggestions(currentRun);
        await loadIdentityEdits(currentRun);
    } catch (error) {
        console.error('Error undoing identity edit:', error);
        setIdentityStatus(`Failed to undo last edit: ${error.message}`, true);
    }
}

async function undoIdentityEditByOperation(encodedOpId) {
    if (!currentRun) return;

    let opId = '';
    try {
        opId = decodeURIComponent(encodedOpId || '');
    } catch (error) {
        opId = String(encodedOpId || '');
    }
    if (!opId) {
        setIdentityStatus('Operation id is required for targeted undo', true);
        return;
    }

    try {
        const response = await fetch(`/api/runs/${currentRun}/identity_review/actions/undo/${encodeURIComponent(opId)}`, {
            method: 'POST'
        });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        const data = await response.json();
        const reverted = Number(data.reverted_count ?? 0);
        const deleted = Number(data.deleted_count ?? 0);
        const failed = Number(data.failed_count ?? 0);
        setIdentityStatus(`Undo ${opId} complete: reverted ${reverted}, deleted ${deleted}, failed ${failed}`);

        selectedAssignmentTrackIds.clear();
        clearRecomputePreview();
        await loadIdentityReview(currentRun);
        await loadIdentitySuggestions(currentRun);
        await loadIdentityEdits(currentRun);
    } catch (error) {
        console.error('Error undoing identity edit by operation:', error);
        setIdentityStatus(`Failed targeted undo (${opId}): ${error.message}`, true);
    }
}

// Get events filtered by the current eventFilterMode
function _isEventTypeShot(eventType) {
    return eventType === 'shot' || eventType === 'shot_on_target' || eventType === 'shot_off_target';
}

function getFilteredEvents() {
    if (!events || events.length === 0) return [];
    return events.filter(ev => {
        if (eventFilterMode === 'all') return true;
        if (eventFilterMode === 'shot') return _isEventTypeShot(ev.event_type);
        if (eventFilterMode === 'goal') return ev.event_type === 'goal';
        return (ev.status || 'pending') === eventFilterMode;
    });
}

// Render events list
function renderEvents() {
    // Update filter counts
    const counts = { all: 0, shot: 0, goal: 0, pending: 0, confirmed: 0, rejected: 0 };
    (events || []).forEach(ev => {
        counts.all++;
        if (_isEventTypeShot(ev.event_type)) counts.shot++;
        if (ev.event_type === 'goal') counts.goal++;
        const st = ev.status || 'pending';
        if (st === 'pending') counts.pending++;
        if (st === 'confirmed') counts.confirmed++;
        if (st === 'rejected') counts.rejected++;
    });
    const countIds = { all: 'filterCountAll', shot: 'filterCountShot', goal: 'filterCountGoal', pending: 'filterCountPending', confirmed: 'filterCountConfirmed', rejected: 'filterCountRejected' };
    Object.entries(countIds).forEach(([key, id]) => {
        const el = document.getElementById(id);
        if (el) el.textContent = counts[key];
    });

    if (!events || events.length === 0) {
        eventsList.innerHTML = '<p class="loading">No events detected</p>';
        return;
    }

    const filteredEvents = getFilteredEvents();

    if (filteredEvents.length === 0) {
        eventsList.innerHTML = '<p class="loading">No events match this filter</p>';
        return;
    }

    const pageEnd = (eventPage + 1) * EVENT_PAGE_SIZE;
    const visibleEvents = filteredEvents.slice(0, pageEnd);
    const remaining = filteredEvents.length - visibleEvents.length;

    let html = '';
    visibleEvents.forEach((event, index) => {
        const time = formatTime(event.timestamp);
        const confidence = event.confidence.toFixed(2);
        const confidenceClass = event.confidence >= 0.7 ? 'high' : '';

        // Status and source info
        const status = event.status || 'pending';
        const source = event.source || 'auto';
        const eventId = event.id || '';

        // Build status badge
        const statusBadge = `<span class="status-badge ${status}">${status}</span>`;

        let details = '';
        if (event.metadata) {
            if (_isEventTypeShot(event.event_type) && event.metadata.speed) {
                details += `Speed: ${event.metadata.speed.toFixed(1)}px/f`;
            }
            if (event.metadata.target_goal) {
                details += ` • Target: ${event.metadata.target_goal}`;
            }
            if (event.metadata.goal_region) {
                details += ` • Region: ${event.metadata.goal_region}`;
            }
        }

        // Add user notes if present
        if (event.user_notes) {
            details += details ? ` • Note: ${event.user_notes}` : `Note: ${event.user_notes}`;
        }

        // Build action buttons based on status and source
        let actionButtons = '';
        if (status === 'pending') {
            actionButtons = `
                <div class="event-actions">
                    <button class="action-btn approve" onclick="showInlineNotes('${eventId}', 'confirm', event)" title="Confirm">Confirm</button>
                    <button class="action-btn reject" onclick="showInlineNotes('${eventId}', 'reject', event)" title="Reject">Reject</button>
                </div>
                <div class="event-inline-notes" id="notes-${eventId}" style="display:none">
                    <input type="text" id="notesInput-${eventId}" placeholder="Notes (optional)..." onkeydown="if(event.key==='Enter'){event.stopPropagation();submitInlineNotes('${eventId}')}">
                    <button class="action-btn approve" onclick="submitInlineNotes('${eventId}', event)">Submit</button>
                    <button class="action-btn" style="background:var(--border);color:var(--text)" onclick="event.stopPropagation();hideInlineNotes('${eventId}')">Cancel</button>
                </div>
            `;
        } else if (source === 'manual') {
            actionButtons = `
                <div class="event-actions">
                    <button class="action-btn delete" onclick="deleteManualEvent('${eventId}', event)" title="Delete">Delete</button>
                </div>
            `;
        }

        // CSS classes for the event item
        const itemClasses = [
            'event-item',
            event.event_type,
            status,
            source === 'manual' ? 'manual' : ''
        ].filter(Boolean).join(' ');

        html += `
            <div class="${itemClasses}" data-event-timestamp="${event.timestamp}" onclick="seekToEvent(${event.timestamp})">
                <div class="event-time">
                    ${event.event_type.toUpperCase()} at ${time}
                    <span class="confidence ${confidenceClass}">${confidence}</span>
                    ${statusBadge}
                </div>
                <div class="event-details">${details}</div>
                ${actionButtons}
            </div>
        `;
    });

    if (remaining > 0) {
        html += `<button class="show-more-btn" onclick="eventPage++;renderEvents()">Show more (${remaining} remaining)</button>`;
    }

    eventsList.innerHTML = html;
    lastHighlightedEventIdx = -1;
}

// Inline notes for confirm/reject
let pendingInlineAction = null;
let pendingInlineEventId = null;

function showInlineNotes(eventId, action, clickEvent) {
    if (clickEvent) clickEvent.stopPropagation();
    pendingInlineAction = action;
    pendingInlineEventId = eventId;
    const container = document.getElementById(`notes-${eventId}`);
    if (container) {
        container.dataset.action = action;
        container.style.display = 'flex';
        const input = document.getElementById(`notesInput-${eventId}`);
        if (input) input.focus();
    }
}

function hideInlineNotes(eventId) {
    const container = document.getElementById(`notes-${eventId}`);
    if (container) container.style.display = 'none';
    pendingInlineAction = null;
    pendingInlineEventId = null;
}

async function submitInlineNotes(eventId, clickEvent) {
    if (clickEvent) clickEvent.stopPropagation();
    const container = document.getElementById(`notes-${eventId}`);
    const input = document.getElementById(`notesInput-${eventId}`);
    const notes = input ? input.value : '';
    // Prefer data-action on container (survives async timing issues)
    const action = (container && container.dataset.action) || pendingInlineAction;
    pendingInlineAction = null;
    pendingInlineEventId = null;

    if (action === 'confirm') {
        await doConfirmEvent(eventId, notes);
    } else if (action === 'reject') {
        await doRejectEvent(eventId, notes);
    }
}

// Render timeline markers
function renderTimelineMarkers() {
    const duration = videoPlayer.duration || 1;

    // Clear existing markers
    const existingMarkers = timelineBar.querySelectorAll('.timeline-marker');
    existingMarkers.forEach(marker => marker.remove());

    events.forEach((event, index) => {
        const marker = document.createElement('div');
        marker.className = `timeline-marker ${event.event_type}`;
        marker.style.left = `${(event.timestamp / duration) * 100}%`;
        marker.dataset.eventIndex = index;

        // Create tooltip
        const tooltip = document.createElement('div');
        tooltip.className = 'timeline-tooltip';
        const status = event.status || 'pending';
        const conf = event.confidence != null ? (event.confidence * 100).toFixed(0) + '%' : '';
        tooltip.textContent = `${event.event_type.toUpperCase()} ${formatTime(event.timestamp)} ${conf} [${status}]`;
        marker.appendChild(tooltip);

        marker.onclick = (e) => {
            e.stopPropagation();
            seekToEvent(event.timestamp);
        };
        timelineBar.appendChild(marker);
    });
}

// Setup video player
function setupVideoPlayer() {
    const scrubber = document.getElementById('timelineScrubber');
    const hoverTooltip = document.getElementById('timelineHoverTooltip');

    videoPlayer.setAttribute('playsinline', '');
    videoPlayer.setAttribute('webkit-playsinline', '');

    // Update progress bar
    videoPlayer.addEventListener('timeupdate', () => {
        if (videoPlayer.duration) {
            const progress = (videoPlayer.currentTime / videoPlayer.duration) * 100;
            timelineProgress.style.width = `${progress}%`;
            if (scrubber && !timelineDragging) scrubber.style.left = `${progress}%`;
        }

        // Update playback time display
        const timeDisplay = document.getElementById('playbackTimeDisplay');
        if (timeDisplay) {
            timeDisplay.textContent = `${formatTime(videoPlayer.currentTime)} / ${formatTime(videoPlayer.duration || 0)}`;
        }

        if (activeSegmentEndTime !== null && videoPlayer.currentTime >= activeSegmentEndTime) {
            videoPlayer.pause();
            activeSegmentEndTime = null;
        }

        enforceSpeedrunPlayback();

        // Highlight current event
        highlightCurrentEvent(videoPlayer.currentTime);

        // Ensure tracks are loaded for upcoming frames
        ensureTracksLoaded();

        // Render overlay for current frame
        renderOverlay();
    });

    // Add markers when video loads + update time labels
    videoPlayer.addEventListener('loadedmetadata', () => {
        renderTimelineMarkers();
        setupCanvas();
        const startLabel = document.getElementById('timelineLabelStart');
        const endLabel = document.getElementById('timelineLabelEnd');
        if (startLabel) startLabel.textContent = '0:00';
        if (endLabel) endLabel.textContent = formatTime(videoPlayer.duration || 0);
    });

    // Resize canvas when video size changes
    videoPlayer.addEventListener('resize', setupCanvas);
    window.addEventListener('resize', setupCanvas);

    // Hover tooltip on timeline
    if (hoverTooltip) {
        timelineBar.addEventListener('mousemove', (e) => {
            if (timelineDragging) return;
            const rect = timelineBar.getBoundingClientRect();
            const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
            const pct = x / rect.width;
            const time = pct * (videoPlayer.duration || 0);
            hoverTooltip.textContent = formatTime(time);
            hoverTooltip.style.left = `${x}px`;
            hoverTooltip.style.opacity = '1';
        });

        timelineBar.addEventListener('mouseleave', () => {
            hoverTooltip.style.opacity = '0';
        });
    }

    // Click timeline to seek
    timelineBar.addEventListener('click', (e) => {
        if (e.target.closest('.timeline-marker')) return;
        const rect = timelineBar.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percentage = x / rect.width;
        activeSegmentEndTime = null;
        videoPlayer.currentTime = percentage * videoPlayer.duration;
    });

    timelineBar.addEventListener('touchend', (e) => {
        if (timelineDragging) return;
        if (e.target && e.target.closest && e.target.closest('.timeline-marker')) return;
        const rect = timelineBar.getBoundingClientRect();
        const touch = e.changedTouches && e.changedTouches[0];
        if (!touch) return;
        const x = Math.max(0, Math.min(rect.width, touch.clientX - rect.left));
        const percentage = x / rect.width;
        activeSegmentEndTime = null;
        videoPlayer.currentTime = percentage * (videoPlayer.duration || 0);
    }, { passive: true });

    // Drag-to-scrub
    if (scrubber) {
        const startDrag = (e) => {
            e.preventDefault();
            timelineDragging = true;
            const onMove = (me) => {
                const rect = timelineBar.getBoundingClientRect();
                const x = Math.max(0, Math.min(rect.width, (me.clientX || me.touches?.[0]?.clientX || 0) - rect.left));
                const pct = x / rect.width;
                scrubber.style.left = `${pct * 100}%`;
                timelineProgress.style.width = `${pct * 100}%`;
                videoPlayer.currentTime = pct * (videoPlayer.duration || 0);
            };
            const onEnd = () => {
                timelineDragging = false;
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onEnd);
                document.removeEventListener('touchmove', onMove);
                document.removeEventListener('touchend', onEnd);
            };
            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onEnd);
            document.addEventListener('touchmove', onMove);
            document.addEventListener('touchend', onEnd);
        };
        scrubber.addEventListener('mousedown', startDrag);
        scrubber.addEventListener('touchstart', startDrag);
    }
}

// Helper: seek then play once the video is ready at the new position.
// Uses the 'seeked' event so the frame is decoded before playback starts,
// which avoids the visible buffering / stall spinner on large files.
function _seekAndPlay(time) {
    videoPlayer.currentTime = time;
    videoPlayer.play().catch(() => {});
}

// Seek to event time
function seekToEvent(timestamp) {
    activeSegmentEndTime = null;
    _seekAndPlay(timestamp);
    updateUrlHash(timestamp);
}

// Play a per-player highlight segment directly in the main player.
function playPlayerSegment(startTime, endTime) {
    if (speedrunState.enabled) {
        disableSpeedrunMode();
    }
    hideClipModal();
    activeSegmentEndTime = Number.isFinite(endTime) ? endTime : null;
    _seekAndPlay(Math.max(0, startTime || 0));
}

function playPlayerClip(playerId, encodedSegmentId, clickEvent) {
    if (clickEvent) {
        clickEvent.stopPropagation();
    }
    if (!currentRun || !clipModal || !clipPlayer) {
        return;
    }

    const clipUrl = `/api/runs/${encodeURIComponent(currentRun)}/player_reels/${playerId}/segments/${encodedSegmentId}/clip`;
    if (clipModalTitle) {
        clipModalTitle.textContent = `Player ${playerId} Clip`;
    }
    clipPlayer.src = clipUrl;
    clipModal.style.display = 'block';
    clipPlayer.play().catch(() => {
        // User gesture requirement may block autoplay.
    });
}

function hideClipModal() {
    if (!clipModal || !clipPlayer) {
        return;
    }
    clipModal.style.display = 'none';
    clipPlayer.pause();
    clipPlayer.removeAttribute('src');
    clipPlayer.load();
}

// Format time as MM:SS or H:MM:SS
function formatTime(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return '0:00';
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    if (hrs > 0) {
        return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Load metadata for run
async function loadMetadata(runName) {
    try {
        const response = await fetchWithRetry(`/api/runs/${runName}/metadata`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        videoMetadata = data.video_metadata;
        runSummaryData = data.summary || null;
        renderRunSummary();
    } catch (error) {
        console.error('Error loading metadata:', error);
    }
}

// --- Run Summary Panel ---
function renderRunSummary() {
    const bar = document.getElementById('runSummaryBar');
    if (!bar) return;

    if (!runSummaryData && !currentRun) {
        bar.classList.remove('visible');
        return;
    }

    let items = [];
    if (currentRun) items.push(`<span class="run-summary-item"><span class="run-summary-value">${escapeHtml(currentRun)}</span></span>`);

    if (runSummaryData) {
        const s = runSummaryData;
        const vid = s.video || {};
        if (vid.duration_seconds) items.push(`<span class="run-summary-item">Duration: <span class="run-summary-value">${(vid.duration_seconds / 60).toFixed(1)}min</span></span>`);
        if (vid.resolution) items.push(`<span class="run-summary-item">Res: <span class="run-summary-value">${vid.resolution.width}x${vid.resolution.height}</span></span>`);

        const counts = s.counts || {};
        if (counts.tracks_unique) items.push(`<span class="run-summary-item">Tracks: <span class="run-summary-value">${counts.tracks_unique}</span></span>`);
        if (counts.events_total) items.push(`<span class="run-summary-item">Events: <span class="run-summary-value">${counts.events_total}</span></span>`);
        if (counts.shots) items.push(`<span class="run-summary-item">Shots: <span class="run-summary-value">${counts.shots}</span></span>`);
        if (counts.goals) items.push(`<span class="run-summary-item">Goals: <span class="run-summary-value">${counts.goals}</span></span>`);
        if (counts.detections_total) items.push(`<span class="run-summary-item">Detections: <span class="run-summary-value">${counts.detections_total.toLocaleString()}</span></span>`);
    }

    bar.innerHTML = items.join('');
    bar.classList.toggle('visible', items.length > 0);
}

// --- Team Analytics ---
async function loadMatchStats(runName) {
    if (!runName) return;
    try {
        const response = await fetchWithRetry(`/api/runs/${runName}/match_stats`);
        if (response.status === 404) {
            matchStatsData = null;
            renderMatchStats();
            return;
        }
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        matchStatsData = await response.json();
        renderMatchStats();
    } catch (error) {
        console.error('Error loading match stats:', error);
        matchStatsData = null;
        renderMatchStats();
    }
}

function _asNumber(value, fallback = 0) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : fallback;
}

function _formatTeamLabel(label) {
    if (!label) return 'Unknown';
    const normalized = String(label).trim();
    if (!normalized) return 'Unknown';
    if (normalized === 'ours') return 'Ours';
    if (normalized === 'opponent') return 'Opponent';
    if (normalized === 'unknown') return 'Unknown';
    return normalized.replace(/_/g, ' ');
}

function _formatMatchMetricValue(value, kind = 'count') {
    const numeric = _asNumber(value, 0);
    if (kind === 'percent') {
        return `${(numeric * 100).toFixed(1)}%`;
    }
    return `${Math.round(numeric)}`;
}

function _pickComparisonTeams(teamsObj) {
    const entries = Object.entries(teamsObj || {});
    if (entries.length === 0) {
        return [];
    }

    const lookup = new Map(entries);
    if (lookup.has('ours') && lookup.has('opponent')) {
        return [['ours', lookup.get('ours') || {}], ['opponent', lookup.get('opponent') || {}]];
    }

    const known = entries.filter(([teamName]) => teamName !== 'unknown');
    if (known.length >= 2) {
        return known.slice(0, 2);
    }
    if (known.length === 1) {
        const secondary = entries.find(([teamName]) => teamName !== known[0][0]) || ['unknown', {}];
        return [known[0], secondary];
    }

    if (entries.length === 1) {
        return [entries[0], ['unknown', {}]];
    }

    return entries.slice(0, 2);
}

function renderMatchStats() {
    const container = document.getElementById('matchStatsContent');
    if (!container) return;

    if (!matchStatsData) {
        container.innerHTML = '<p class="loading">No match stats data loaded</p>';
        return;
    }

    const teams = matchStatsData.teams || {};
    const pair = _pickComparisonTeams(teams);
    if (pair.length < 2) {
        container.innerHTML = '<p class="loading">Not enough team data for comparison</p>';
        return;
    }

    const [leftTeamName, leftTeamStats = {}] = pair[0];
    const [rightTeamName, rightTeamStats = {}] = pair[1];

    const metrics = [
        { key: 'shots', label: 'Shots', kind: 'count' },
        { key: 'goals', label: 'Goals', kind: 'count' },
        { key: 'passes', label: 'Passes', kind: 'count' },
        { key: 'set_pieces', label: 'Set Pieces', kind: 'count' },
        { key: 'possession_share', label: 'Possession', kind: 'percent' },
    ];

    const comparisonRows = metrics.map((metric) => {
        const leftValue = _asNumber(leftTeamStats[metric.key], 0);
        const rightValue = _asNumber(rightTeamStats[metric.key], 0);
        const total = leftValue + rightValue;
        const leftWidth = total > 0 ? (leftValue / total) * 100 : 50;
        const rightWidth = 100 - leftWidth;

        return `
            <div class="match-stats-row">
                <div class="match-stats-value match-stats-left">${_formatMatchMetricValue(leftValue, metric.kind)}</div>
                <div class="match-stats-metric">
                    <div class="match-stats-metric-label">${escapeHtml(metric.label)}</div>
                    <div class="match-stats-bar">
                        <div class="match-stats-bar-left" style="width:${leftWidth.toFixed(1)}%"></div>
                        <div class="match-stats-bar-right" style="width:${rightWidth.toFixed(1)}%"></div>
                    </div>
                </div>
                <div class="match-stats-value match-stats-right">${_formatMatchMetricValue(rightValue, metric.kind)}</div>
            </div>
        `;
    }).join('');

    const totals = matchStatsData.totals || {};
    const summary = matchStatsData.summary || {};

    container.innerHTML = `
        <div class="match-stats-header">
            <span class="match-stats-team match-stats-team-left">${escapeHtml(_formatTeamLabel(leftTeamName))}</span>
            <span class="match-stats-vs">vs</span>
            <span class="match-stats-team match-stats-team-right">${escapeHtml(_formatTeamLabel(rightTeamName))}</span>
        </div>
        <div class="match-stats-rows">
            ${comparisonRows}
        </div>
        <div class="match-stats-totals">
            <span>Totals: ${_formatMatchMetricValue(totals.shots)} shots, ${_formatMatchMetricValue(totals.goals)} goals, ${_formatMatchMetricValue(totals.passes)} passes, ${_formatMatchMetricValue(totals.set_pieces)} set pieces</span>
            <span>Events processed: ${_formatMatchMetricValue(summary.events_processed)} (${_formatMatchMetricValue(summary.events_without_team)} unattributed)</span>
        </div>
    `;
}

async function loadTeamAnalytics(runName) {
    if (!runName) return;
    try {
        const response = await fetchWithRetry(`/api/runs/${runName}/team_analytics`);
        if (response.status === 404) {
            teamAnalyticsData = null;
            renderTeamAnalytics();
            return;
        }
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        teamAnalyticsData = await response.json();
        renderTeamAnalytics();
    } catch (error) {
        console.error('Error loading team analytics:', error);
        teamAnalyticsData = null;
        renderTeamAnalytics();
    }
}

function renderTeamAnalytics() {
    const container = document.getElementById('teamAnalyticsContent');
    if (!container) return;

    if (!teamAnalyticsData) {
        container.innerHTML = '<p class="loading">No team analytics data loaded</p>';
        return;
    }

    const possession = teamAnalyticsData.possession || {};
    const territory = teamAnalyticsData.territory || {};
    const passNetwork = teamAnalyticsData.pass_network || {};
    const pressing = teamAnalyticsData.pressing || {};

    let html = '';

    // Possession bar
    const teams = possession.teams || {};
    const teamEntries = Object.entries(teams);
    if (teamEntries.length > 0) {
        html += '<div style="font-size:0.78rem;color:var(--muted);margin-bottom:0.25rem;">Possession</div>';
        html += '<div class="analytics-possession-bar">';
        const colors = ['#1565d8', '#e05555', '#888'];
        teamEntries.forEach(([team, data], i) => {
            const share = (Number(data.share || 0) * 100).toFixed(1);
            const seconds = Number(data.seconds || 0).toFixed(0);
            html += `<div style="width:${share}%;background:${colors[i % colors.length]}">${escapeHtml(team)} ${share}% (${seconds}s)</div>`;
        });
        html += '</div>';
    }

    // Pressing KPIs
    const pressingTeams = pressing.teams || {};
    const pressingEntries = Object.entries(pressingTeams);
    if (pressingEntries.length > 0) {
        html += '<div style="font-size:0.78rem;color:var(--muted);margin-top:0.75rem;margin-bottom:0.25rem;">Pressing</div>';
        html += '<div class="analytics-kpi-row">';
        pressingEntries.forEach(([team, data]) => {
            html += `
                <div class="analytics-kpi">
                    <div class="analytics-kpi-label">${escapeHtml(team)} Pressure</div>
                    <div class="analytics-kpi-value">${Number(data.avg_pressure_score || 0).toFixed(2)}</div>
                </div>
                <div class="analytics-kpi">
                    <div class="analytics-kpi-label">${escapeHtml(team)} High Press</div>
                    <div class="analytics-kpi-value">${(Number(data.high_press_rate || 0) * 100).toFixed(1)}%</div>
                </div>
            `;
        });
        html += '</div>';
    }

    // Territory grid
    const territoryTeams = territory.teams || {};
    const territoryEntries = Object.entries(territoryTeams);
    if (territoryEntries.length >= 2) {
        html += '<div style="font-size:0.78rem;color:var(--muted);margin-top:0.75rem;margin-bottom:0.25rem;">Territory</div>';
        html += '<div class="analytics-territory-grid">';
        const xBins = territory.x_bins || ['left', 'center', 'right'];
        const yBins = territory.y_bins || ['top', 'middle', 'bottom'];
        const teamA = territoryEntries[0];
        const teamB = territoryEntries[1];
        const teamAZones = teamA[1].x_zone_control_share || {};
        const teamBZones = teamB[1].x_zone_control_share || {};

        yBins.forEach(yBin => {
            xBins.forEach(xBin => {
                const aShare = Number(teamAZones[xBin] || 0);
                const bShare = Number(teamBZones[xBin] || 0);
                const total = aShare + bShare || 1;
                const aPct = ((aShare / total) * 100).toFixed(0);
                const color = aShare > bShare ? 'rgba(21,101,216,0.3)' : 'rgba(224,85,85,0.3)';
                html += `<div class="analytics-territory-cell" style="background:${color}">${aPct}%</div>`;
            });
        });
        html += '</div>';
        html += `<div style="font-size:0.68rem;color:var(--muted);">${escapeHtml(teamA[0])} (blue) vs ${escapeHtml(teamB[0])} (red) zone control</div>`;
    }

    // Pass network top edges
    const topEdges = passNetwork.top_edges || [];
    if (topEdges.length > 0) {
        html += '<div style="font-size:0.78rem;color:var(--muted);margin-top:0.75rem;margin-bottom:0.25rem;">Pass Network (Top Edges)</div>';
        html += '<table class="analytics-pass-table"><tr><th>From</th><th>To</th><th>Passes</th><th>Team</th></tr>';
        topEdges.slice(0, 8).forEach(edge => {
            html += `<tr><td>${escapeHtml(edge.from_label || edge.from_track_id)}</td><td>${escapeHtml(edge.to_label || edge.to_track_id)}</td><td>${edge.count || 0}</td><td>${escapeHtml(edge.team || '')}</td></tr>`;
        });
        html += '</table>';
    }

    if (!html) html = '<p class="loading">No analytics content to display</p>';
    container.innerHTML = html;
}

function _optionalNumber(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
}

function _buildVisualizationFilters() {
    const filters = {
        type: visualizationTypeSelect ? visualizationTypeSelect.value : visualizationState.type,
        teamId: visualizationTeamFilter ? String(visualizationTeamFilter.value || '').trim() : '',
        playerId: visualizationPlayerFilter ? _optionalNumber(visualizationPlayerFilter.value) : null,
        minConfidence: visualizationMinConfidence ? _optionalNumber(visualizationMinConfidence.value) : 0.0,
        includePoints: visualizationIncludePoints ? Boolean(visualizationIncludePoints.checked) : false
    };

    if (!filters.type) {
        filters.type = 'pass_map';
    }
    if (!Number.isFinite(filters.minConfidence)) {
        filters.minConfidence = 0.0;
    }
    return filters;
}

function _formatVisualizationTeamLabel(teamId) {
    if (!teamId) return 'Unknown';
    if (teamId === 'ours') return 'Ours';
    if (teamId === 'opponent') return 'Opponent';
    if (teamId === 'unknown') return 'Unknown';
    return String(teamId).replace(/_/g, ' ');
}

function renderVisualizationPanel() {
    if (!visualizationContent) return;

    if (!currentRun) {
        visualizationContent.innerHTML = '<p class="loading">Select a run to load visualization maps</p>';
        return;
    }

    if (!visualizationData) {
        const message = visualizationError || 'No visualization map available for this run';
        visualizationContent.innerHTML = `<p class="loading">${escapeHtml(message)}</p>`;
        return;
    }

    const payload = visualizationData.payload || {};
    const metadata = visualizationData.metadata || {};
    const totals = payload.totals || {};
    const encoded = payload.image_png_base64 || '';
    const imageTag = encoded
        ? `<img class="visualization-image" src="data:image/png;base64,${encoded}" alt="${escapeHtml(visualizationData.title || 'Visualization map')}">`
        : '<p class="loading">Visualization image unavailable</p>';

    const type = String(visualizationData.visualization_type || visualizationState.type || 'pass_map');
    const summaryItems = [];
    if (type === 'pass_map') {
        summaryItems.push(`Passes: ${Number(totals.passes || 0)}`);
        summaryItems.push(`Edges: ${Number(totals.edges || 0)}`);
        summaryItems.push(`Nodes: ${Number(totals.nodes || 0)}`);
    } else if (type === 'shot_map') {
        summaryItems.push(`Shots: ${Number(totals.shots || 0)}`);
        summaryItems.push(`Goals: ${Number(totals.goals || 0)}`);
        summaryItems.push(`Teams: ${Number(totals.teams || 0)}`);
    } else if (type === 'heat_map') {
        summaryItems.push(`Samples: ${Number(totals.samples || 0)}`);
        summaryItems.push(`Teams: ${Number(totals.teams || 0)}`);
        summaryItems.push(`Tracks: ${Number(totals.tracks || 0)}`);
    } else if (type === 'momentum') {
        summaryItems.push(`Windows: ${Number(totals.windows || 0)}`);
        summaryItems.push(`Duration: ${Number(totals.duration_seconds || 0).toFixed(0)}s`);
    } else if (type === 'pass_strings') {
        summaryItems.push(`Chains: ${Number(totals.chains || 0)}`);
        summaryItems.push(`Max length: ${Number(totals.max_chain_length || 0)}`);
    } else if (type === 'radial_chart') {
        summaryItems.push(`Metrics: ${Number(totals.metrics || 0)}`);
        summaryItems.push(`Teams: ${Number(totals.teams || 0)}`);
    } else if (type === 'progress_chart') {
        summaryItems.push(`Windows: ${Number(totals.windows || 0)}`);
        summaryItems.push(`Duration: ${Number(totals.duration_seconds || 0).toFixed(0)}s`);
    } else if (type === 'tactical_map') {
        summaryItems.push(`Players: ${Number(totals.players || 0)}`);
        summaryItems.push(`Teams: ${Number(totals.teams || 0)}`);
    } else {
        summaryItems.push(`Samples: ${Number(totals.samples || 0)}`);
        summaryItems.push(`Teams: ${Number(totals.teams || 0)}`);
        summaryItems.push(`Tracks: ${Number(totals.tracks || 0)}`);
    }

    const teams = Array.isArray(metadata.teams) ? metadata.teams : [];
    const teamPills = teams.length > 0
        ? teams.map((team) => `<span class="viz-pill">${escapeHtml(_formatVisualizationTeamLabel(team))}</span>`).join('')
        : '<span class="viz-pill">No team labels</span>';

    visualizationContent.innerHTML = `
        <div class="visualization-meta">
            <div class="visualization-summary">${summaryItems.map((item) => `<span class="viz-pill">${escapeHtml(item)}</span>`).join('')}</div>
            <div class="visualization-summary">${teamPills}</div>
        </div>
        <div class="visualization-image-wrap">
            ${imageTag}
        </div>
    `;
}

async function loadVisualization(runName) {
    if (!runName || !visualizationContent) return;

    const filters = _buildVisualizationFilters();
    visualizationState = {
        ...visualizationState,
        ...filters
    };

    const KNOWN_VIZ_TYPES = ['pass_map', 'shot_map', 'heat_map', 'tactical_map',
        'momentum', 'pass_strings', 'radial_chart', 'progress_chart'];
    const endpointType = KNOWN_VIZ_TYPES.includes(filters.type) ? filters.type : 'pass_map';
    const params = new URLSearchParams();
    if (filters.teamId) params.set('team_id', filters.teamId);
    if (filters.playerId != null) params.set('player_id', String(Math.round(filters.playerId)));
    if (filters.minConfidence != null) params.set('min_confidence', String(Math.max(0, Math.min(1, filters.minConfidence))));
    params.set('canvas_width', '900');
    params.set('canvas_height', '560');

    if (endpointType === 'tactical_map' || endpointType === 'heat_map') {
        params.set('include_points', filters.includePoints ? 'true' : 'false');
    }

    visualizationError = '';
    visualizationData = null;
    visualizationContent.innerHTML = '<p class="loading">Loading visualization map...</p>';

    try {
        const response = await fetchWithRetry(`/api/runs/${runName}/visualizations/${endpointType}?${params.toString()}`);
        if (response.status === 404) {
            visualizationError = 'Visualization not available for this run';
            visualizationData = null;
            renderVisualizationPanel();
            return;
        }
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        visualizationData = await response.json();
        visualizationError = '';
        renderVisualizationPanel();
    } catch (error) {
        console.error('Error loading visualization map:', error);
        visualizationData = null;
        visualizationError = `Failed loading map: ${error.message}`;
        renderVisualizationPanel();
    }
}

function onVisualizationControlsChanged() {
    if (!currentRun) return;
    loadVisualization(currentRun);
}

// --- Formation / Lineup ---
async function loadLineup(runName) {
    if (!runName) return;
    try {
        const resp = await fetch(`/api/runs/${runName}/lineup`);
        if (!resp.ok) return;
        const data = await resp.json();
        const formEl = document.getElementById('lineupFormation');
        const notesEl = document.getElementById('lineupNotes');
        if (formEl) formEl.value = data.formation || '';
        if (notesEl) notesEl.value = data.notes || '';
        const content = document.getElementById('lineupContent');
        if (content) {
            const count = Array.isArray(data.players) ? data.players.length : 0;
            content.innerHTML = `<p style="font-size:0.8rem;color:#aaa;">${count} player(s) in lineup</p>`;
        }
    } catch (e) { console.error('Error loading lineup:', e); }
}

async function saveLineup() {
    if (!currentRun) return;
    const formation = document.getElementById('lineupFormation')?.value || '';
    const notes = document.getElementById('lineupNotes')?.value || '';
    try {
        await fetch(`/api/runs/${currentRun}/lineup`, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({formation, players: [], notes}),
        });
    } catch (e) { console.error('Error saving lineup:', e); }
}

// --- Coach Notes ---
async function loadCoachNotes(runName) {
    if (!runName) return;
    const content = document.getElementById('coachNotesContent');
    if (!content) return;
    try {
        const resp = await fetch(`/api/runs/${runName}/notes`);
        if (!resp.ok) { content.innerHTML = '<p class="loading">No notes</p>'; return; }
        const data = await resp.json();
        const notes = data.notes || [];
        if (!notes.length) { content.innerHTML = '<p class="loading">No notes yet</p>'; return; }
        content.innerHTML = notes.map(n => `
            <div style="display:flex;justify-content:space-between;align-items:center;padding:0.3rem 0;border-bottom:1px solid #333;font-size:0.8rem;">
                <div><span style="color:#888;">[${escapeHtml(n.category || 'general')}]</span> ${escapeHtml(n.text)}</div>
                <button class="identity-btn" style="font-size:0.7rem;padding:0.15rem 0.5rem;" onclick="deleteCoachNote('${n.id}')">X</button>
            </div>
        `).join('');
    } catch (e) { console.error('Error loading notes:', e); content.innerHTML = '<p class="loading">Error loading notes</p>'; }
}

async function addCoachNote() {
    if (!currentRun) return;
    const textEl = document.getElementById('coachNoteText');
    const catEl = document.getElementById('coachNoteCategory');
    const text = textEl?.value?.trim();
    if (!text) return;
    try {
        await fetch(`/api/runs/${currentRun}/notes`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text, category: catEl?.value || 'general'}),
        });
        if (textEl) textEl.value = '';
        loadCoachNotes(currentRun);
    } catch (e) { console.error('Error adding note:', e); }
}

async function deleteCoachNote(noteId) {
    if (!currentRun) return;
    try {
        await fetch(`/api/runs/${currentRun}/notes/${noteId}`, {method: 'DELETE'});
        loadCoachNotes(currentRun);
    } catch (e) { console.error('Error deleting note:', e); }
}

// --- Player Spotlight Config ---
async function loadSpotlightConfig(runName) {
    if (!runName) return;
    try {
        const resp = await fetch(`/api/runs/${runName}/spotlight_config`);
        if (!resp.ok) return;
        const data = await resp.json();
        const el = (id) => document.getElementById(id);
        if (el('spotlightBallDistance')) el('spotlightBallDistance').value = data.ball_distance_threshold || 140;
        if (el('spotlightTimeOnBall')) el('spotlightTimeOnBall').value = data.time_on_ball_seconds || 1.5;
        if (el('spotlightPreBuffer')) el('spotlightPreBuffer').value = data.pre_buffer_seconds || 3.0;
        if (el('spotlightPostBuffer')) el('spotlightPostBuffer').value = data.post_buffer_seconds || 3.0;
    } catch (e) { console.error('Error loading spotlight config:', e); }
}

async function saveSpotlightConfig() {
    if (!currentRun) return;
    const el = (id) => document.getElementById(id);
    try {
        await fetch(`/api/runs/${currentRun}/spotlight_config`, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                ball_distance_threshold: parseFloat(el('spotlightBallDistance')?.value) || 140,
                time_on_ball_seconds: parseFloat(el('spotlightTimeOnBall')?.value) || 1.5,
                pre_buffer_seconds: parseFloat(el('spotlightPreBuffer')?.value) || 3.0,
                post_buffer_seconds: parseFloat(el('spotlightPostBuffer')?.value) || 3.0,
            }),
        });
    } catch (e) { console.error('Error saving spotlight config:', e); }
}

// Load tracks for overlay rendering (progressive loading in windows)
async function loadTracks(runName) {
    try {
        // Load initial window of tracks (first 30 seconds = ~900 frames)
        // Note: tracks typically start at frame 20, not 0
        const fps = videoMetadata?.fps || 30;
        const initialFrames = Math.floor(fps * 30); // 30 seconds

        await loadTracksWindow(runName, 20, initialFrames);
        lastLoadedFrame = initialFrames;

        console.log(`Initial tracks loaded (20-${initialFrames}). Will load more as needed during playback.`);
    } catch (error) {
        console.error('Error loading tracks:', error);
        tracks = [];
        tracksByFrame = {};
    }
}

// Load tracks for a specific frame window
async function loadTracksWindow(runName, frameStart, frameEnd) {
    try {
        const response = await fetch(`/api/runs/${runName}/tracks?frame_start=${frameStart}&frame_end=${frameEnd}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const data = await response.json();

        // Filter out tracks with invalid coordinates (null values from NaN)
        const rawTracks = Array.isArray(data.tracks) ? data.tracks : [];
        const validTracks = rawTracks.filter(track =>
            track.x1 != null && track.y1 != null &&
            track.x2 != null && track.y2 != null
        );

        // Add tracks to the global array
        tracks = tracks.concat(validTracks);

        // Index tracks by frame for fast lookup
        validTracks.forEach(track => {
            const frameIdx = track.frame_idx;
            if (!tracksByFrame[frameIdx]) {
                tracksByFrame[frameIdx] = [];
            }
            tracksByFrame[frameIdx].push(track);
        });

        console.log(`Loaded frames ${frameStart}-${frameEnd}: ${validTracks.length} valid tracks (${data.tracks.length - validTracks.length} filtered)`);
    } catch (error) {
        console.error(`Error loading tracks window ${frameStart}-${frameEnd}:`, error);
    }
}

// Preload tracks ahead of current playback position
let lastLoadedFrame = 0;
async function ensureTracksLoaded() {
    if (!videoMetadata || !currentRun) return;

    const fps = videoMetadata.fps || 30;
    const currentFrame = Math.floor(videoPlayer.currentTime * fps);

    // Load 30 seconds ahead
    const windowSize = Math.floor(fps * 30);
    const nextWindowEnd = currentFrame + windowSize;

    // Check if we need to load more
    if (nextWindowEnd > lastLoadedFrame) {
        const loadStart = lastLoadedFrame + 1;
        const loadEnd = nextWindowEnd;
        await loadTracksWindow(currentRun, loadStart, loadEnd);
        lastLoadedFrame = loadEnd;
    }
}

// Setup canvas size to match video
function setupCanvas() {
    if (!videoPlayer || !overlayCanvas) return;
    if (!videoPlayer.videoWidth || !videoPlayer.videoHeight) return;
    const rect = videoPlayer.getBoundingClientRect();
    overlayCanvas.width = videoPlayer.videoWidth;
    overlayCanvas.height = videoPlayer.videoHeight;
    overlayCanvas.style.width = `${rect.width}px`;
    overlayCanvas.style.height = `${rect.height}px`;
}

// Render overlay on canvas
function renderOverlay() {
    if (!videoMetadata || !tracks.length) {
        return;
    }

    // Clear canvas
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    // Calculate current frame
    const fps = videoMetadata.fps || 30;
    const currentFrame = Math.floor(videoPlayer.currentTime * fps);

    // Get tracks for current frame
    const currentTracks = tracksByFrame[currentFrame] || [];

    // Debug logging (only when tracks exist for this frame)
    if (currentTracks.length > 0 && currentFrame % 30 === 0) {
        console.log(`Frame ${currentFrame}: rendering ${currentTracks.length} tracks`);
    }

    // Build track history for trails
    const trackHistory = {};
    if (overlaySettings.layers.trails) {
        const startFrame = Math.max(0, currentFrame - overlaySettings.trailLength);
        for (let f = startFrame; f <= currentFrame; f++) {
            const frameTracks = tracksByFrame[f] || [];
            frameTracks.forEach(track => {
                if (!trackHistory[track.track_id]) {
                    trackHistory[track.track_id] = [];
                }
                const cx = (track.x1 + track.x2) / 2;
                const cy = (track.y1 + track.y2) / 2;
                trackHistory[track.track_id].push({ x: cx, y: cy });
            });
        }
    }

    // Draw trails first (so they appear behind boxes)
    if (overlaySettings.layers.trails) {
        drawTrails(trackHistory);
    }

    // Draw bounding boxes and labels
    if (overlaySettings.layers.boxes || overlaySettings.layers.labels) {
        drawDetections(currentTracks);
    }
}

// Draw track trails
function drawTrails(trackHistory) {
    Object.entries(trackHistory).forEach(([trackId, points]) => {
        if (points.length < 2) return;

        const isDark = document.documentElement.classList.contains('dark');
        ctx.strokeStyle = isDark ? '#00e5ff' : '#00bcd4';
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.6;

        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);

        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i].x, points[i].y);
        }

        ctx.stroke();
        ctx.globalAlpha = 1.0;
    });
}

// Draw bounding boxes and labels
function drawDetections(currentTracks) {
    currentTracks.forEach(track => {
        const { x1, y1, x2, y2, object_type, team_name, track_id, confidence } = track;

        // Choose color based on object type and team
        let color;
        if (object_type === 'ball') {
            color = overlaySettings.colors.ball;
        } else if (object_type === 'player') {
            if (team_name === 'team_A') {
                color = overlaySettings.colors.team_A;
            } else if (team_name === 'team_B') {
                color = overlaySettings.colors.team_B;
            } else {
                color = '#808080'; // Gray for unknown team
            }
        } else {
            color = '#808080'; // Gray for unknown type
        }

        // Draw bounding box
        if (overlaySettings.layers.boxes) {
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        }

        // Draw label
        if (overlaySettings.layers.labels) {
            const label = `${object_type} ID:${track_id}`;
            const fontSize = 14;
            ctx.font = `${fontSize}px Arial`;
            const metrics = ctx.measureText(label);
            const textWidth = metrics.width;
            const textHeight = fontSize;

            // Draw label background
            ctx.fillStyle = color;
            ctx.fillRect(x1, y1 - textHeight - 4, textWidth + 8, textHeight + 4);

            // Draw label text
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, x1 + 4, y1 - 4);
        }
    });
}

// Toggle layer visibility
function toggleLayer(layer) {
    overlaySettings.layers[layer] = !overlaySettings.layers[layer];

    // Update button state
    const btn = document.getElementById(`toggle${layer.charAt(0).toUpperCase() + layer.slice(1)}`);
    if (btn) {
        if (overlaySettings.layers[layer]) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    }

    // Re-render
    renderOverlay();
}

// Toggle between original and overlay video
function toggleOriginalVideo() {
    showingOriginal = !showingOriginal;

    const btn = document.getElementById('toggleOriginal');
    const currentTime = videoPlayer.currentTime;

    if (showingOriginal) {
        // Show overlay video (baked) — use HLS if available
        loadVideoSource(currentRun, true, currentTime);
        btn.classList.add('active');
        btn.textContent = 'Show Original';
        // Hide canvas overlay when showing baked overlay
        overlayCanvas.style.display = 'none';
    } else {
        // Show original video with dynamic overlay
        loadVideoSource(currentRun, false, currentTime);
        btn.classList.remove('active');
        btn.textContent = 'Show Overlay Video';
        // Show canvas overlay
        overlayCanvas.style.display = 'block';
    }
}

// Update team color
function updateTeamColor(team, color) {
    overlaySettings.colors[team] = color;
    renderOverlay();
}

// Update ball color
function updateBallColor(color) {
    overlaySettings.colors.ball = color;
    renderOverlay();
}

// Event confirmation functions

async function confirmEvent(eventId, clickEvent) {
    if (clickEvent) clickEvent.stopPropagation();
    showInlineNotes(eventId, 'confirm', clickEvent);
}

async function doConfirmEvent(eventId, notes) {
    try {
        const response = await fetchWithRetry(`/api/runs/${currentRun}/events/${eventId}/confirm`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ notes: notes || '' })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        showToast('Event confirmed', 'success');
        await loadEvents(currentRun);
    } catch (error) {
        console.error('Error confirming event:', error);
        showToast('Failed to confirm event', 'error');
    }
}

async function rejectEvent(eventId, clickEvent) {
    if (clickEvent) clickEvent.stopPropagation();
    showInlineNotes(eventId, 'reject', clickEvent);
}

async function doRejectEvent(eventId, notes) {
    try {
        const response = await fetchWithRetry(`/api/runs/${currentRun}/events/${eventId}/reject`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ notes: notes || '' })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        showToast('Event rejected', 'success');
        await loadEvents(currentRun);
    } catch (error) {
        console.error('Error rejecting event:', error);
        showToast('Failed to reject event', 'error');
    }
}

async function deleteManualEvent(eventId, clickEvent) {
    if (clickEvent) clickEvent.stopPropagation();

    const confirmed = await showConfirmModal('Are you sure you want to delete this event?');
    if (!confirmed) return;

    try {
        const response = await fetchWithRetry(`/api/runs/${currentRun}/events/${eventId}`, {
            method: 'DELETE'
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        showToast('Event deleted', 'success');
        await loadEvents(currentRun);
    } catch (error) {
        console.error('Error deleting event:', error);
        showToast('Failed to delete event', 'error');
    }
}

function showAddEventModal() {
    if (!currentRun) {
        showToast('Please select a run first', 'error');
        return;
    }

    const modal = document.getElementById('addEventModal');
    const timestampInput = document.getElementById('eventTimestamp');
    if (!modal || !timestampInput) return;

    // Set current video position
    const currentTime = videoPlayer.currentTime || 0;
    timestampInput.value = formatTime(currentTime) + ` (${currentTime.toFixed(2)}s)`;
    timestampInput.dataset.timestamp = currentTime;

    // Calculate frame index
    const fps = videoMetadata?.fps || 30;
    const frameIdx = Math.floor(currentTime * fps);
    timestampInput.dataset.frameIdx = frameIdx;

    // Reset form
    document.getElementById('eventType').value = 'shot';
    document.getElementById('eventNotes').value = '';

    modal.style.display = 'block';
}

function hideAddEventModal() {
    const modal = document.getElementById('addEventModal');
    if (modal) modal.style.display = 'none';
}

async function submitAddEvent(formEvent) {
    formEvent.preventDefault();

    const timestampInput = document.getElementById('eventTimestamp');
    const eventType = document.getElementById('eventType').value;
    const notes = document.getElementById('eventNotes').value;

    const timestamp = parseFloat(timestampInput.dataset.timestamp);
    const frameIdx = parseInt(timestampInput.dataset.frameIdx);

    try {
        const response = await fetch(`/api/runs/${currentRun}/events`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                event_type: eventType,
                timestamp: timestamp,
                frame_idx: frameIdx,
                notes: notes,
                metadata: {}
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || `HTTP ${response.status}`);
        }

        hideAddEventModal();
        showToast('Event added', 'success');
        await loadEvents(currentRun);
    } catch (error) {
        console.error('Error adding event:', error);
        showToast('Failed to add event: ' + error.message, 'error');
    }
}

// Close modal when clicking outside of it
window.addEventListener('click', (event) => {
    const modal = document.getElementById('addEventModal');
    if (event.target === modal) {
        hideAddEventModal();
    }

    if (clipModal && event.target === clipModal) {
        hideClipModal();
    }

    const shortcutsModal = document.getElementById('shortcutsModal');
    if (shortcutsModal && event.target === shortcutsModal) {
        hideShortcutsModal();
    }

    const confirmModalEl = document.getElementById('confirmModal');
    if (confirmModalEl && event.target === confirmModalEl) {
        hideConfirmModal();
    }
});

// --- Playback Controls ---

function stepFrame(direction) {
    videoPlayer.pause();
    const fps = videoMetadata?.fps || 30;
    const step = direction / fps;
    videoPlayer.currentTime = Math.max(0, Math.min(videoPlayer.duration || 0, videoPlayer.currentTime + step));
}

function setPlaybackSpeed(speed) {
    currentPlaybackSpeed = speed;
    videoPlayer.playbackRate = speed;
    document.querySelectorAll('.speed-btn').forEach(btn => {
        const btnSpeed = parseFloat(btn.textContent);
        if (btnSpeed === speed) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
}

// --- Event Filtering ---

function setEventFilter(mode) {
    eventFilterMode = mode;
    eventPage = 0;
    document.querySelectorAll('.event-filters .filter-btn').forEach(btn => {
        if (btn.dataset.filter === mode) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
    renderEvents();
    updateUrlHash();
}

function resetEventFilterButtons() {
    document.querySelectorAll('.event-filters .filter-btn').forEach(btn => {
        if (btn.dataset.filter === 'all') {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });
}

// --- Current Event Highlighting ---

function highlightCurrentEvent(currentTime) {
    if (!events || events.length === 0) {
        lastHighlightedEventIdx = -1;
        return;
    }

    // Search only filtered events (same set currently rendered)
    const filtered = getFilteredEvents();

    // Find closest filtered event within 3 seconds
    let closestIdx = -1;
    let closestDist = Infinity;
    filtered.forEach((ev, idx) => {
        const dist = Math.abs(ev.timestamp - currentTime);
        if (dist < 3 && dist < closestDist) {
            closestDist = dist;
            closestIdx = idx;
        }
    });

    // Skip DOM work if the highlighted event hasn't changed
    if (closestIdx === lastHighlightedEventIdx) return;
    lastHighlightedEventIdx = closestIdx;

    // Remove all current highlights
    document.querySelectorAll('.event-item.current').forEach(el => el.classList.remove('current'));
    document.querySelectorAll('.timeline-marker.current').forEach(el => el.classList.remove('current'));

    if (closestIdx >= 0) {
        const closestEvent = filtered[closestIdx];

        // Highlight timeline marker by matching the full events array index
        const fullIdx = events.indexOf(closestEvent);
        if (fullIdx >= 0) {
            const marker = timelineBar.querySelector(`.timeline-marker[data-event-index="${fullIdx}"]`);
            if (marker) marker.classList.add('current');
        }

        // Highlight event item via data attribute
        const ts = String(closestEvent.timestamp);
        const item = eventsList.querySelector(`.event-item[data-event-timestamp="${ts}"]`);
        if (item) item.classList.add('current');
    }
}

// --- Dark Mode ---

function toggleTheme() {
    document.documentElement.classList.toggle('dark');
    const isDark = document.documentElement.classList.contains('dark');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    const themeBtn = document.getElementById('themeToggle');
    if (themeBtn) {
        themeBtn.textContent = isDark ? 'Light Mode' : 'Dark Mode';
    }
}

// --- Deep Linking ---

function updateUrlHash(time) {
    if (!currentRun) return;
    let hash = `#run=${encodeURIComponent(currentRun)}`;
    if (time != null && Number.isFinite(time)) {
        hash += `&time=${time.toFixed(2)}`;
    }
    if (eventFilterMode && eventFilterMode !== 'all') {
        hash += `&filter=${encodeURIComponent(eventFilterMode)}`;
    }
    if (playerReelFilters.team && playerReelFilters.team !== 'all') {
        hash += `&reelTeam=${encodeURIComponent(playerReelFilters.team)}`;
    }
    if (playerReelFilters.sortBy && playerReelFilters.sortBy !== 'best_score_desc') {
        hash += `&reelSort=${encodeURIComponent(playerReelFilters.sortBy)}`;
    }
    history.replaceState(null, '', hash);
}

function parseUrlHash() {
    const hash = location.hash.replace(/^#/, '');
    const params = {};
    hash.split('&').forEach(part => {
        const [key, value] = part.split('=');
        if (key && value) {
            params[key] = decodeURIComponent(value);
        }
    });
    return params;
}

async function restoreFromHash() {
    await loadRuns();
    const params = parseUrlHash();
    if (params.run) {
        const runExists = allRunsData.some(r => r.name === params.run);
        if (!runExists) {
            console.warn(`Deep-link run "${params.run}" not found in available runs. Clearing hash.`);
            history.replaceState(null, '', location.pathname);
            return;
        }
        // Restore filter state before loading
        if (params.filter) {
            eventFilterMode = params.filter;
        }
        if (params.reelTeam && reelTeamFilter) {
            reelTeamFilter.value = params.reelTeam;
            playerReelFilters.team = params.reelTeam;
        }
        if (params.reelSort && reelSortBy) {
            reelSortBy.value = params.reelSort;
            playerReelFilters.sortBy = params.reelSort;
        }
        await loadRun(params.run);
        showView('matchAnalysisView');
        // Restore event filter buttons after render
        if (params.filter) {
            setEventFilter(params.filter);
        }
        if (params.time) {
            const t = parseFloat(params.time);
            if (Number.isFinite(t)) {
                videoPlayer.currentTime = t;
            }
        }
    }
}

// Handle browser back/forward navigation
window.addEventListener('hashchange', () => {
    const params = parseUrlHash();
    if (params.run) {
        const runExists = allRunsData.some(r => r.name === params.run);
        if (runExists && params.run !== currentRun) {
            showView('matchAnalysisView');
            loadRun(params.run).then(() => {
                if (params.time) {
                    const t = parseFloat(params.time);
                    if (Number.isFinite(t)) {
                        videoPlayer.currentTime = t;
                    }
                }
            });
        }
    }
});

// --- Shortcuts Modal ---

function showShortcutsModal() {
    const modal = document.getElementById('shortcutsModal');
    if (modal) modal.style.display = 'block';
}

function hideShortcutsModal() {
    const modal = document.getElementById('shortcutsModal');
    if (modal) modal.style.display = 'none';
}

// --- Keyboard Shortcuts ---

const SPEED_STEPS = [0.25, 0.5, 1, 1.5, 2];

document.addEventListener('keydown', (e) => {
    // Skip if user is typing in an input field (except Escape)
    const tag = (e.target.tagName || '').toLowerCase();
    const isTyping = tag === 'input' || tag === 'textarea' || tag === 'select' || e.target.isContentEditable;

    // Escape closes any open modal
    if (e.key === 'Escape') {
        const addEventModal = document.getElementById('addEventModal');
        const shortcutsModalEl = document.getElementById('shortcutsModal');
        const clipModalEl = document.getElementById('clipModal');
        const confirmModalEl = document.getElementById('confirmModal');

        if (addEventModal && addEventModal.style.display === 'block') { hideAddEventModal(); return; }
        if (clipModalEl && clipModalEl.style.display === 'block') { hideClipModal(); return; }
        if (shortcutsModalEl && shortcutsModalEl.style.display === 'block') { hideShortcutsModal(); return; }
        if (confirmModalEl && confirmModalEl.style.display === 'block') { hideConfirmModal(); return; }
        return;
    }

    if (isTyping) return;

    // Skip if any modal is open (except shortcuts modal for closing)
    const addEventModal = document.getElementById('addEventModal');
    const shortcutsModalEl = document.getElementById('shortcutsModal');
    const clipModalEl = document.getElementById('clipModal');
    const confirmModalEl = document.getElementById('confirmModal');

    const addEventOpen = addEventModal && addEventModal.style.display === 'block';
    const clipOpen = clipModalEl && clipModalEl.style.display === 'block';
    const confirmOpen = confirmModalEl && confirmModalEl.style.display === 'block';

    if (addEventOpen || clipOpen || confirmOpen) return;

    // ? key - toggle shortcuts
    if (e.key === '?') {
        if (shortcutsModalEl && shortcutsModalEl.style.display === 'block') {
            hideShortcutsModal();
        } else {
            showShortcutsModal();
        }
        return;
    }

    // Close shortcuts modal on any other key if open
    if (shortcutsModalEl && shortcutsModalEl.style.display === 'block') {
        return;
    }

    switch (e.key) {
        case ' ':
            e.preventDefault();
            if (videoPlayer.paused) {
                videoPlayer.play();
            } else {
                videoPlayer.pause();
            }
            break;

        case 'ArrowLeft':
            e.preventDefault();
            if (e.shiftKey) {
                stepFrame(-1);
            } else {
                videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime - 5);
            }
            break;

        case 'ArrowRight':
            e.preventDefault();
            if (e.shiftKey) {
                stepFrame(1);
            } else {
                videoPlayer.currentTime = Math.min(videoPlayer.duration || 0, videoPlayer.currentTime + 5);
            }
            break;

        case 'j':
            videoPlayer.currentTime = Math.max(0, videoPlayer.currentTime - 10);
            break;

        case 'k':
            if (videoPlayer.paused) {
                videoPlayer.play();
            } else {
                videoPlayer.pause();
            }
            break;

        case 'l':
            videoPlayer.currentTime = Math.min(videoPlayer.duration || 0, videoPlayer.currentTime + 10);
            break;

        case 'f':
            toggleFullscreen();
            break;

        case 'x':
            toggleSpeedrunMode();
            break;

        case 'v':
            toggleViewerLayout();
            break;

        case 'm':
            showAddEventModal();
            break;

        case '[': {
            const curIdx = SPEED_STEPS.indexOf(currentPlaybackSpeed);
            if (curIdx > 0) {
                setPlaybackSpeed(SPEED_STEPS[curIdx - 1]);
            }
            break;
        }

        case ']': {
            const curIdx = SPEED_STEPS.indexOf(currentPlaybackSpeed);
            if (curIdx < SPEED_STEPS.length - 1) {
                setPlaybackSpeed(SPEED_STEPS[curIdx + 1]);
            }
            break;
        }

        default:
            // Number keys 1-9: jump to Nth visible event
            if (e.key >= '1' && e.key <= '9') {
                const n = parseInt(e.key);
                const filtered = getFilteredEvents();
                if (n <= filtered.length) {
                    seekToEvent(filtered[n - 1].timestamp);
                }
            }
            break;
    }
});

// --- Onboarding Card ---
function renderOnboardingCard() {
    const card = document.getElementById('onboardingCard');
    if (!card) return;

    const dismissed = localStorage.getItem('onboarding_dismissed') === '1';
    if (dismissed || allRunsData.length > 0) {
        card.innerHTML = '';
        return;
    }

    card.innerHTML = `
        <div class="onboarding-card">
            <h3>Getting Started</h3>
            <div class="onboarding-steps">
                <div class="onboarding-step">
                    <div class="onboarding-step-number">1</div>
                    <p>Enter a video path in <a href="#" onclick="showView('pipelineStudioView');return false;" style="color:var(--accent);text-decoration:underline;">Pipeline Studio</a></p>
                </div>
                <div class="onboarding-step">
                    <div class="onboarding-step-number">2</div>
                    <p>Click "Queue Analysis" to start processing</p>
                </div>
                <div class="onboarding-step">
                    <div class="onboarding-step-number">3</div>
                    <p>Review match events here, then use the menu for Player Reels and Season Trends</p>
                </div>
            </div>
            <button class="onboarding-dismiss" onclick="dismissOnboarding()">Dismiss</button>
        </div>
    `;
}

function dismissOnboarding() {
    localStorage.setItem('onboarding_dismissed', '1');
    const card = document.getElementById('onboardingCard');
    if (card) card.innerHTML = '';
}

// ═══════════════════════════════════════════════════════════════════════
// Navigation & View Switching
// ═══════════════════════════════════════════════════════════════════════

function toggleNav() {
    const panel = document.getElementById('navPanel');
    const overlay = document.getElementById('navOverlay');
    const btn = document.getElementById('burgerBtn');
    const isOpen = panel.classList.contains('open');
    panel.classList.toggle('open', !isOpen);
    overlay.classList.toggle('open', !isOpen);
    btn.classList.toggle('open', !isOpen);
}

function showView(viewName) {
    // Hide all views
    const views = ['pipelineStudioView', 'matchAnalysisView', 'playerReelsView', 'seasonTrendsView', 'teamManagerView', 'playerManagerView'];
    views.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = id === viewName ? '' : 'none';
    });

    // Update nav active state
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.view === viewName);
    });

    // Close nav if opened from burger
    const panel = document.getElementById('navPanel');
    if (panel && panel.classList.contains('open')) {
        toggleNav();
    }

    // Load data when switching to views
    if (viewName === 'pipelineStudioView') {
        loadPipelineJobs(true);
    } else if (viewName === 'teamManagerView') {
        loadTeams();
    } else if (viewName === 'playerManagerView') {
        loadPlayersManager();
    } else if (viewName === 'playerReelsView' || viewName === 'seasonTrendsView') {
        updateCrossViewContexts();
        if (!currentRun) {
            showToast('Select a run in Match Analysis to load this workspace.', 'info');
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Team Manager
// ═══════════════════════════════════════════════════════════════════════

async function loadTeams() {
    try {
        const response = await fetchWithRetry('/api/teams');
        if (!response.ok) throw new Error('Failed to load teams');
        const data = await response.json();
        teamsData = data.teams || [];
        renderTeamsList();
    } catch (err) {
        console.error('Error loading teams:', err);
        const el = document.getElementById('teamsList');
        if (el) el.innerHTML = '<p style="color:var(--muted);">Could not load teams.</p>';
    }
}

function renderTeamsList() {
    const el = document.getElementById('teamsList');
    if (!el) return;
    if (teamsData.length === 0) {
        el.innerHTML = '<p style="color:var(--muted);padding:1rem;">No teams yet. Click "Create Team" to get started.</p>';
        return;
    }
    el.innerHTML = teamsData.map(team => {
        const kitsHtml = (team.kits || []).map(k => {
            const bg = k.color_hex || '#ccc';
            return `<div style="text-align:center;">
                <div class="kit-swatch" style="background:${bg};" title="${k.kit_type}"></div>
                <div class="kit-swatch-label">${k.kit_type}</div>
            </div>`;
        }).join('');
        const initials = (team.short_name || team.name || '?').slice(0, 3).toUpperCase();
        const logoHtml = team.logo_path
            ? `<img class="team-logo-sm" src="/api/teams/${team.team_id}/logo" alt="logo">`
            : `<div class="team-logo-initials">${escapeHtml(initials)}</div>`;
        return `<div class="team-card" onclick="openTeamDetail(${team.team_id})">
            <div class="team-card-header">
                ${logoHtml}
                <div style="flex:1;min-width:0;">
                    <h3>${escapeHtml(team.name)}</h3>
                    <span class="team-card-meta">${team.short_name ? escapeHtml(team.short_name) : ''}</span>
                </div>
            </div>
            <div class="team-card-meta">${team.player_count} player${team.player_count !== 1 ? 's' : ''}</div>
            <div class="kit-swatches">${kitsHtml || '<span style="color:var(--muted);font-size:0.8rem;">No kits</span>'}</div>
        </div>`;
    }).join('');
}

// escapeHtml is defined earlier in this file (line ~1828) with full entity escaping

function showCreateTeamForm() {
    const form = document.getElementById('createTeamForm');
    if (form) form.style.display = 'flex';
    const input = document.getElementById('newTeamName');
    if (input) { input.value = ''; input.focus(); }
}

function hideCreateTeamForm() {
    const form = document.getElementById('createTeamForm');
    if (form) form.style.display = 'none';
}

async function createTeam() {
    const nameInput = document.getElementById('newTeamName');
    const shortInput = document.getElementById('newTeamShortName');
    const name = nameInput ? nameInput.value.trim() : '';
    if (!name) { showToast('Team name is required', 'warning'); return; }

    try {
        const response = await fetch('/api/teams', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, short_name: shortInput ? shortInput.value.trim() || null : null })
        });
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Failed to create team');
        }
        showToast(`Team "${name}" created`, 'success');
        hideCreateTeamForm();
        await loadTeams();
        loadPipelineTeamSelectors();
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function openTeamDetail(teamId) {
    currentTeamId = teamId;
    const el = document.getElementById('teamDetail');
    if (!el) return;
    el.style.display = '';
    el.innerHTML = '<p class="loading">Loading team details...</p>';

    try {
        const [teamResponse, playersResponse] = await Promise.all([
            fetchWithRetry(`/api/teams/${teamId}`),
            fetchWithRetry('/api/players')
        ]);
        if (!teamResponse.ok) throw new Error('Failed to load team');
        const team = await teamResponse.json();
        const allPlayers = playersResponse.ok ? ((await playersResponse.json()).players || []) : [];
        renderTeamDetail(team, allPlayers);
    } catch (err) {
        el.innerHTML = `<p style="color:var(--warning);">${err.message}</p>`;
    }
}

function renderTeamDetail(team, allPlayers = []) {
    const el = document.getElementById('teamDetail');
    if (!el) return;

    const kitTypes = ['home', 'away', 'third'];
    const kitsMap = {};
    (team.kits || []).forEach(k => kitsMap[k.kit_type] = k);

    const kitsHtml = kitTypes.map(type => {
        const kit = kitsMap[type];
        const thumbHtml = kit && kit.image_path
            ? `<img class="kit-thumb" src="/api/teams/${team.team_id}/kits/${type}/image" alt="${type} kit">`
            : `<div class="kit-thumb-placeholder">+</div>`;
        const swatchHtml = kit && kit.color_hex
            ? `<div class="kit-swatch" style="background:${kit.color_hex};" title="Primary"></div>` +
              (kit.secondary_color_hex ? `<div class="kit-swatch" style="background:${kit.secondary_color_hex};" title="Secondary"></div>` : '')
            : '';
        return `<div class="kit-slot">
            ${thumbHtml}
            <div style="flex:1;">
                <strong style="text-transform:capitalize;">${type}</strong>
                <div class="kit-swatches" style="margin-top:0.25rem;">${swatchHtml}</div>
            </div>
            <input type="file" accept="image/*" id="kitUpload_${type}" style="display:none;" onchange="uploadKit(${team.team_id}, '${type}', this)">
            <button class="identity-btn" onclick="document.getElementById('kitUpload_${type}').click()">Upload</button>
            ${kit ? `<button class="identity-btn" onclick="deleteKit(${team.team_id}, '${type}')">Remove</button>` : ''}
        </div>`;
    }).join('');

    // Build "Add Existing Player" dropdown from unlinked players
    const teamPlayerIds = new Set((team.players || []).map(p => p.player_id));
    const unlinkedPlayers = allPlayers.filter(p => !p.team_id && !teamPlayerIds.has(p.player_id));
    let addExistingHtml;
    if (unlinkedPlayers.length > 0) {
        const options = unlinkedPlayers.map(p =>
            `<option value="${p.player_id}">${escapeHtml(p.name || 'Unnamed')} ${p.jersey_number ? '#' + p.jersey_number : ''}</option>`
        ).join('');
        addExistingHtml = `<div style="display:flex;gap:0.4rem;align-items:center;margin-bottom:0.5rem;">
            <select id="rosterAddExisting" class="form-input" style="flex:1;">${options}</select>
            <button class="identity-btn" onclick="addExistingPlayerToTeam()">Add</button>
        </div>`;
    } else {
        addExistingHtml = `<p style="color:var(--muted);font-size:0.8rem;margin-bottom:0.5rem;">No unlinked players available.</p>`;
    }

    // "Create New Player" inline form
    const createNewHtml = `<div style="display:flex;gap:0.4rem;align-items:center;margin-bottom:0.5rem;">
        <input type="text" id="rosterNewName" class="form-input" placeholder="Name" style="flex:1;">
        <input type="number" id="rosterNewNumber" class="form-input" placeholder="#" style="width:3.5rem;">
        <button class="identity-btn" onclick="createAndAddPlayerToTeam()">Create &amp; Add</button>
    </div>`;

    const playersHtml = (team.players || []).map(p =>
        `<div style="display:flex;align-items:center;padding:0.35rem 0;border-bottom:1px solid var(--border);gap:0.4rem;">
            <span style="flex:1;min-width:0;">${escapeHtml(p.name || 'Unnamed')}</span>
            <span style="width:3.5rem;text-align:center;color:var(--muted);">${p.jersey_number != null ? '#' + p.jersey_number : ''}</span>
            <button class="identity-btn" style="color:var(--warning);white-space:nowrap;" onclick="removePlayerFromTeam(${p.player_id})">Remove</button>
        </div>`
    ).join('') || '<p style="color:var(--muted);font-size:0.85rem;">No players linked to this team.</p>';

    const logoHtml = team.logo_path
        ? `<img class="team-logo-lg" src="/api/teams/${team.team_id}/logo" alt="logo">`
        : `<div class="team-logo-placeholder" onclick="document.getElementById('logoUpload').click()">+</div>`;

    el.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1rem;">
            <div style="display:flex;gap:1rem;align-items:center;">
                <div style="text-align:center;">
                    ${logoHtml}
                    <input type="file" accept="image/*" id="logoUpload" style="display:none;" onchange="uploadTeamLogo(${team.team_id}, this)">
                    <div style="display:flex;gap:0.25rem;margin-top:0.3rem;">
                        <button class="identity-btn" style="font-size:0.7rem;" onclick="document.getElementById('logoUpload').click()">Upload</button>
                        ${team.logo_path ? `<button class="identity-btn" style="font-size:0.7rem;color:var(--warning);" onclick="deleteTeamLogo(${team.team_id})">Remove</button>` : ''}
                    </div>
                </div>
                <div>
                    <input type="text" class="form-input" value="${escapeHtml(team.name)}" id="teamDetailName" onblur="updateTeamField(${team.team_id}, 'name')" style="font-size:1.1rem;font-weight:700;">
                    <input type="text" class="form-input" value="${escapeHtml(team.short_name || '')}" id="teamDetailShortName" onblur="updateTeamField(${team.team_id}, 'short_name')" placeholder="Short name" style="margin-top:0.3rem;">
                </div>
            </div>
            <button class="identity-btn" style="color:var(--warning);" onclick="deleteTeam(${team.team_id})">Delete Team</button>
        </div>
        <h3 style="margin-bottom:0.5rem;">Kits</h3>
        ${kitsHtml}
        <h3 style="margin-top:1rem;margin-bottom:0.5rem;">Roster</h3>
        ${playersHtml}
        <div style="margin-top:0.75rem;padding-top:0.5rem;border-top:1px solid var(--border);">
            <p style="color:var(--muted);font-size:0.78rem;margin-bottom:0.4rem;">Add player</p>
            ${addExistingHtml}
            ${createNewHtml}
        </div>
        <a href="#" onclick="showView('playerManagerView');return false;" style="display:inline-block;margin-top:0.5rem;color:var(--accent);font-size:0.85rem;">Manage all players &rarr;</a>
        <button class="identity-btn" style="margin-top:0.5rem;" onclick="closeTeamDetail()">Close</button>
    `;
}

function closeTeamDetail() {
    const el = document.getElementById('teamDetail');
    if (el) el.style.display = 'none';
    currentTeamId = null;
}

async function updateTeamField(teamId, field) {
    const nameEl = document.getElementById('teamDetailName');
    const shortEl = document.getElementById('teamDetailShortName');
    const body = {};
    if (field === 'name' && nameEl) body.name = nameEl.value.trim();
    if (field === 'short_name' && shortEl) body.short_name = shortEl.value.trim();

    try {
        await fetch(`/api/teams/${teamId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        await loadTeams();
        loadPipelineTeamSelectors();
    } catch (err) {
        showToast('Failed to update team', 'error');
    }
}

async function deleteTeam(teamId) {
    const confirmed = await showConfirmModal('Delete this team? This will remove all kits and unlink all players.');
    if (!confirmed) return;

    try {
        const response = await fetch(`/api/teams/${teamId}`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Failed to delete');
        showToast('Team deleted', 'success');
        closeTeamDetail();
        await loadTeams();
        loadPipelineTeamSelectors();
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function uploadKit(teamId, kitType, inputEl) {
    const file = inputEl.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    showLoadingBar();
    try {
        const response = await fetch(`/api/teams/${teamId}/kits/${kitType}`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error('Upload failed');
        showToast('Kit uploaded & colors extracted', 'success');
        await openTeamDetail(teamId);
        await loadTeams();
        loadPipelineTeamSelectors();
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        hideLoadingBar();
        inputEl.value = '';
    }
}

async function deleteKit(teamId, kitType) {
    try {
        await fetch(`/api/teams/${teamId}/kits/${kitType}`, { method: 'DELETE' });
        showToast('Kit removed', 'success');
        await openTeamDetail(teamId);
        await loadTeams();
        loadPipelineTeamSelectors();
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function uploadTeamLogo(teamId, inputEl) {
    const file = inputEl.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    showLoadingBar();
    try {
        const response = await fetch(`/api/teams/${teamId}/logo`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error('Upload failed');
        showToast('Logo uploaded', 'success');
        await openTeamDetail(teamId);
        await loadTeams();
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        hideLoadingBar();
        inputEl.value = '';
    }
}

async function deleteTeamLogo(teamId) {
    try {
        await fetch(`/api/teams/${teamId}/logo`, { method: 'DELETE' });
        showToast('Logo removed', 'success');
        await openTeamDetail(teamId);
        await loadTeams();
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function removePlayerFromTeam(playerId) {
    try {
        await fetch(`/api/players/${playerId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ team_id: 0 })
        });
        showToast('Player removed from team', 'success');
        if (currentTeamId) await openTeamDetail(currentTeamId);
        await loadTeams();
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function updatePlayerField(playerId, field, inputEl) {
    const newVal = inputEl.value;
    const origVal = inputEl.dataset.orig;
    if (newVal === origVal) return; // nothing changed

    const body = {};
    if (field === 'name') {
        body.name = newVal.trim() || null;
    } else if (field === 'jersey_number') {
        const num = parseInt(newVal, 10);
        body.jersey_number = isNaN(num) ? null : num;
    }
    try {
        const response = await fetch(`/api/players/${playerId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        if (!response.ok) throw new Error('Failed to update player');
        inputEl.dataset.orig = newVal; // update baseline so next blur is a no-op
        showToast('Player updated', 'success');
    } catch (err) {
        showToast(err.message, 'error');
        inputEl.value = origVal; // revert on failure
    }
}

async function addExistingPlayerToTeam() {
    const sel = document.getElementById('rosterAddExisting');
    if (!sel || !sel.value || !currentTeamId) return;
    const playerId = sel.value;
    try {
        const response = await fetch(`/api/players/${playerId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ team_id: currentTeamId })
        });
        if (!response.ok) throw new Error('Failed to add player');
        showToast('Player added to team', 'success');
        await openTeamDetail(currentTeamId);
        await loadTeams();
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function createAndAddPlayerToTeam() {
    if (!currentTeamId) return;
    const nameEl = document.getElementById('rosterNewName');
    const numEl = document.getElementById('rosterNewNumber');
    const name = (nameEl ? nameEl.value.trim() : '');
    const jerseyNumber = numEl ? parseInt(numEl.value, 10) : NaN;
    if (!name && isNaN(jerseyNumber)) {
        showToast('Enter a name or jersey number', 'error');
        return;
    }
    try {
        const createBody = {};
        if (name) createBody.name = name;
        if (!isNaN(jerseyNumber)) createBody.jersey_number = jerseyNumber;
        createBody.team_hint = 'ours';
        const createResp = await fetch('/api/players', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(createBody)
        });
        if (!createResp.ok) throw new Error('Failed to create player');
        const created = await createResp.json();
        const newPlayerId = created.player.player_id;
        const patchResp = await fetch(`/api/players/${newPlayerId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ team_id: currentTeamId })
        });
        if (!patchResp.ok) throw new Error('Player created but failed to link to team');
        showToast(`Player "${name || '#' + jerseyNumber}" created and added`, 'success');
        await openTeamDetail(currentTeamId);
        await loadTeams();
    } catch (err) {
        showToast(err.message, 'error');
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Pipeline Team Selectors
// ═══════════════════════════════════════════════════════════════════════

async function loadPipelineTeamSelectors() {
    try {
        const response = await fetchWithRetry('/api/teams');
        if (!response.ok) return;
        const data = await response.json();
        const teams = data.teams || [];
        populateTeamSelect('pipelineHomeTeam', teams);
        populateTeamSelect('pipelineAwayTeam', teams);
    } catch (err) {
        // Non-fatal
    }
}

function populateTeamSelect(selectId, teams) {
    const sel = document.getElementById(selectId);
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = '<option value="">None</option>' +
        teams.map(t => `<option value="${t.team_id}">${escapeHtml(t.name)}</option>`).join('');
    if (prev) sel.value = prev;
}

function updatePipelineKitSelector(role) {
    const teamSel = document.getElementById(role === 'home' ? 'pipelineHomeTeam' : 'pipelineAwayTeam');
    const colorSpan = document.getElementById(role === 'home' ? 'pipelineHomeColor' : 'pipelineAwayColor');
    if (!teamSel || !colorSpan) return;

    const teamId = teamSel.value;
    if (!teamId) {
        colorSpan.style.background = 'transparent';
        return;
    }

    // Find team in cached data and show primary kit color
    const team = teamsData.find(t => t.team_id == teamId) ||
                 { kits: [] };
    const kitSel = document.getElementById(role === 'home' ? 'pipelineHomeKit' : 'pipelineAwayKit');
    const kitType = kitSel ? kitSel.value : 'home';
    const kit = (team.kits || []).find(k => k.kit_type === kitType);
    colorSpan.style.background = kit && kit.color_hex ? kit.color_hex : 'transparent';
}

// ═══════════════════════════════════════════════════════════════════════
// Run Team Mapping Bar
// ═══════════════════════════════════════════════════════════════════════

async function loadRunTeamMapping(runName) {
    const bar = document.getElementById('teamMappingBar');
    if (!bar) return;

    try {
        const response = await fetchWithRetry(`/api/runs/${runName}/teams`);
        if (!response.ok) { bar.style.display = 'none'; return; }
        const data = await response.json();
        const assocs = data.associations || [];
        if (assocs.length === 0) { bar.style.display = 'none'; return; }

        bar.style.display = 'flex';
        const homeAssoc = assocs.find(a => a.role === 'home');
        const awayAssoc = assocs.find(a => a.role === 'away');

        const badge = (assoc, label) => {
            if (!assoc) return `<span class="team-badge">${label}: ?</span>`;
            const color = assoc.color_hex || '#999';
            const name = assoc.team_name || `Team ${assoc.team_id}`;
            return `<span class="team-badge"><span class="kit-swatch-inline" style="background:${color};"></span> ${label}: ${escapeHtml(name)}</span>`;
        };

        bar.innerHTML = badge(homeAssoc, 'Home') + ' vs ' + badge(awayAssoc, 'Away') +
            `<button class="identity-btn" style="margin-left:auto;" onclick="swapRunTeams('${escapeHtml(runName)}')">Swap Teams</button>`;
    } catch (err) {
        bar.style.display = 'none';
    }
}

async function swapRunTeams(runName) {
    try {
        const response = await fetch(`/api/runs/${runName}/teams/remap`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        if (!response.ok) throw new Error('Swap failed');
        showToast('Teams swapped', 'success');
        await loadRunTeamMapping(runName);
    } catch (err) {
        showToast(err.message, 'error');
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Player Manager View
// ═══════════════════════════════════════════════════════════════════════

let pmAllTeams = [];

async function loadPlayersManager() {
    try {
        const [playersRes, teamsRes] = await Promise.all([
            fetchWithRetry('/api/players'),
            fetchWithRetry('/api/teams')
        ]);
        if (!playersRes.ok || !teamsRes.ok) throw new Error('Failed to load data');
        const playersData = await playersRes.json();
        const teamsData = await teamsRes.json();
        pmAllTeams = teamsData.teams || [];
        renderPlayersManager(playersData.players || []);
    } catch (err) {
        const grid = document.getElementById('playersGrid');
        if (grid) grid.innerHTML = '<p style="color:var(--warning);">Failed to load players.</p>';
    }
}

function showCreatePlayerForm() {
    const form = document.getElementById('createPlayerForm');
    if (form) {
        form.style.display = 'flex';
        populateTeamSelect('mgrPlayerTeam', pmAllTeams);
    }
}

function hideCreatePlayerForm() {
    const form = document.getElementById('createPlayerForm');
    if (form) form.style.display = 'none';
}

async function createPlayerFromManager() {
    const nameEl = document.getElementById('mgrPlayerName');
    const numEl = document.getElementById('newPlayerNumber');
    const teamEl = document.getElementById('mgrPlayerTeam');
    const hintEl = document.getElementById('newPlayerHint');

    const name = nameEl ? nameEl.value.trim() : '';
    const jerseyNumber = numEl ? parseInt(numEl.value, 10) : NaN;
    const teamId = teamEl ? parseInt(teamEl.value, 10) : NaN;
    const teamHint = hintEl ? hintEl.value : '';

    if (!name && isNaN(jerseyNumber)) {
        showToast('Enter a name or jersey number', 'error');
        return;
    }

    try {
        const createBody = {};
        if (name) createBody.name = name;
        if (!isNaN(jerseyNumber)) createBody.jersey_number = jerseyNumber;
        if (teamHint) createBody.team_hint = teamHint;

        const createResp = await fetch('/api/players', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(createBody)
        });
        if (!createResp.ok) throw new Error('Failed to create player');
        const created = await createResp.json();
        const newPlayerId = created.player.player_id;

        if (!isNaN(teamId) && teamId) {
            await fetch(`/api/players/${newPlayerId}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ team_id: teamId })
            });
        }

        showToast('Player created', 'success');
        hideCreatePlayerForm();
        if (nameEl) nameEl.value = '';
        if (numEl) numEl.value = '';
        await loadPlayersManager();
    } catch (err) {
        showToast(err.message, 'error');
    }
}

function renderPlayersManager(players) {
    const grid = document.getElementById('playersGrid');
    if (!grid) return;

    if (players.length === 0) {
        grid.innerHTML = '<p style="color:var(--muted);">No players yet. Click "+ Add Player" to create one.</p>';
        return;
    }

    grid.innerHTML = players.map(p => {
        const photoHtml = p.photo_path
            ? `<img class="player-photo" src="/api/players/${p.player_id}/photo" alt="Photo" onclick="document.getElementById('pmPhotoUpload_${p.player_id}').click()" onerror="this.outerHTML='<div class=\\'player-photo-placeholder\\' onclick=\\'document.getElementById(\\\\'pmPhotoUpload_${p.player_id}\\\\').click()\\'>+</div>'">`
            : `<div class="player-photo-placeholder" onclick="document.getElementById('pmPhotoUpload_${p.player_id}').click()">+</div>`;

        const teamOptions = '<option value="">No team</option>' +
            pmAllTeams.map(t => `<option value="${t.team_id}"${p.team_id === t.team_id ? ' selected' : ''}>${escapeHtml(t.name)}</option>`).join('');

        const hintOptions = ['', 'ours', 'opponent'].map(h =>
            `<option value="${h}"${(p.team_hint || '') === h ? ' selected' : ''}>${h || 'No hint'}</option>`
        ).join('');

        return `<div class="player-card">
            ${photoHtml}
            <input type="file" accept="image/*" id="pmPhotoUpload_${p.player_id}" style="display:none;" onchange="uploadPlayerPhoto(${p.player_id}, this)">
            <div class="player-card-info">
                <div class="field-row">
                    <input type="text" class="roster-inline-input" value="${escapeHtml(p.name || '')}" placeholder="Name"
                        data-orig="${escapeHtml(p.name || '')}"
                        style="flex:1;min-width:0;" onblur="updatePlayerFieldPM(${p.player_id}, 'name', this)">
                    <input type="number" class="roster-inline-input" value="${p.jersey_number != null ? p.jersey_number : ''}" placeholder="#"
                        data-orig="${p.jersey_number != null ? p.jersey_number : ''}"
                        style="width:3.5rem;text-align:center;" onblur="updatePlayerFieldPM(${p.player_id}, 'jersey_number', this)">
                </div>
                <div class="field-row">
                    <select class="form-input" style="flex:1;font-size:0.8rem;padding:0.2rem 0.3rem;" onchange="updatePlayerTeamPM(${p.player_id}, this.value)">
                        ${teamOptions}
                    </select>
                    <select class="form-input" style="width:90px;font-size:0.8rem;padding:0.2rem 0.3rem;" onchange="updatePlayerHintPM(${p.player_id}, this.value)">
                        ${hintOptions}
                    </select>
                </div>
                <div class="player-card-actions">
                    ${p.photo_path ? `<button class="identity-btn" style="font-size:0.75rem;" onclick="deletePlayerPhoto(${p.player_id})">Remove Photo</button>` : ''}
                    <button class="identity-btn" style="font-size:0.75rem;" onclick="document.getElementById('pmTrainUpload_${p.player_id}').click()">Train Face</button>
                    <input type="file" accept="image/*" multiple id="pmTrainUpload_${p.player_id}" style="display:none;" onchange="uploadTrainingImages(${p.player_id}, this)">
                    <span class="embedding-badge" onclick="loadTrainingImages(${p.player_id})" title="Click to view training images">${p.embedding_count > 0 ? p.embedding_count + ' emb' : '0 emb'}</span>
                    <button class="identity-btn" style="font-size:0.75rem;color:var(--warning);" onclick="deletePlayerPM(${p.player_id})">Delete</button>
                </div>
                <div id="trainingGallery_${p.player_id}" class="training-images-gallery" style="display:none;"></div>
            </div>
        </div>`;
    }).join('');
}

async function uploadPlayerPhoto(playerId, inputEl) {
    const file = inputEl.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    showLoadingBar();
    try {
        const response = await fetch(`/api/players/${playerId}/photo`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error('Upload failed');
        showToast('Photo uploaded', 'success');
        await loadPlayersManager();
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        hideLoadingBar();
        inputEl.value = '';
    }
}

async function deletePlayerPhoto(playerId) {
    try {
        const response = await fetch(`/api/players/${playerId}/photo`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Failed to delete photo');
        showToast('Photo removed', 'success');
        await loadPlayersManager();
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function uploadTrainingImages(playerId, inputEl) {
    const files = inputEl.files;
    if (!files || files.length === 0) return;

    const formData = new FormData();
    for (const file of files) {
        formData.append('files', file);
    }

    showLoadingBar();
    try {
        const response = await fetch(`/api/players/${playerId}/training-images`, {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error('Upload failed');
        const data = await response.json();
        const s = data.stats || {};
        showToast(`Processed ${s.total_images_processed || 0} images, ${s.successful_extractions || 0} embeddings generated`, 'success');
        await loadPlayersManager();
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        hideLoadingBar();
        inputEl.value = '';
    }
}

async function loadTrainingImages(playerId) {
    const gallery = document.getElementById(`trainingGallery_${playerId}`);
    if (!gallery) return;

    const isVisible = gallery.style.display !== 'none';
    if (isVisible) {
        gallery.style.display = 'none';
        return;
    }

    try {
        const response = await fetch(`/api/players/${playerId}/training-images`);
        if (!response.ok) throw new Error('Failed to load training images');
        const data = await response.json();

        if (data.count === 0) {
            gallery.innerHTML = '<p style="color:var(--muted);font-size:0.8rem;margin:0.3rem 0;">No training images yet.</p>';
        } else {
            gallery.innerHTML = data.images.map(img =>
                `<div class="training-thumb-wrapper">
                    <img src="${img.url}" alt="${img.filename}" class="training-thumb">
                    <button class="training-thumb-delete" onclick="deleteTrainingImage(${playerId}, '${img.filename}')">&times;</button>
                </div>`
            ).join('');
        }
        gallery.style.display = 'grid';
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function deleteTrainingImage(playerId, filename) {
    showLoadingBar();
    try {
        const response = await fetch(`/api/players/${playerId}/training-images/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });
        if (!response.ok) throw new Error('Failed to delete training image');
        const data = await response.json();
        showToast(`Image removed. ${data.embedding_count} embeddings remaining.`, 'success');
        await loadPlayersManager();
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        hideLoadingBar();
    }
}

async function updatePlayerFieldPM(playerId, field, inputEl) {
    const newVal = inputEl.value;
    const origVal = inputEl.dataset.orig;
    if (newVal === origVal) return;

    const body = {};
    if (field === 'name') {
        body.name = newVal.trim() || null;
    } else if (field === 'jersey_number') {
        const num = parseInt(newVal, 10);
        body.jersey_number = isNaN(num) ? null : num;
    }
    try {
        const response = await fetch(`/api/players/${playerId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        if (!response.ok) throw new Error('Failed to update player');
        inputEl.dataset.orig = newVal;
        showToast('Player updated', 'success');
    } catch (err) {
        showToast(err.message, 'error');
        inputEl.value = origVal;
    }
}

async function updatePlayerTeamPM(playerId, teamIdStr) {
    const teamId = parseInt(teamIdStr, 10);
    try {
        await fetch(`/api/players/${playerId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ team_id: teamId || 0 })
        });
        showToast('Team updated', 'success');
    } catch (err) {
        showToast(err.message, 'error');
        await loadPlayersManager();
    }
}

async function updatePlayerHintPM(playerId, hint) {
    try {
        await fetch(`/api/players/${playerId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ team_hint: hint || null })
        });
        showToast('Hint updated', 'success');
    } catch (err) {
        showToast(err.message, 'error');
        await loadPlayersManager();
    }
}

async function deletePlayerPM(playerId) {
    const confirmed = await showConfirmModal('Delete this player? This will unlink all their appearances.');
    if (!confirmed) return;

    try {
        const response = await fetch(`/api/players/${playerId}`, { method: 'DELETE' });
        if (!response.ok) throw new Error('Failed to delete');
        showToast('Player deleted', 'success');
        await loadPlayersManager();
    } catch (err) {
        showToast(err.message, 'error');
    }
}
