// Global state
let currentRun = null;
let events = [];
let timeline = null;

// DOM elements
const runsList = document.getElementById('runsList');
const viewer = document.getElementById('viewer');
const videoPlayer = document.getElementById('videoPlayer');
const videoSource = document.getElementById('videoSource');
const eventsList = document.getElementById('eventsList');
const scoreDisplay = document.getElementById('scoreDisplay');
const timelineProgress = document.getElementById('timelineProgress');
const timelineBar = document.getElementById('timelineBar');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadRuns();
    setupVideoPlayer();
});

// Load available runs
async function loadRuns() {
    try {
        const response = await fetch('/api/runs');
        const data = await response.json();

        if (data.runs.length === 0) {
            runsList.innerHTML = '<h2>Analysis Runs</h2><p class="loading">No runs found. Run analysis first.</p>';
            return;
        }

        let html = '<h2>Analysis Runs</h2>';
        data.runs.forEach(run => {
            const duration = run.duration ? `${(run.duration / 60).toFixed(1)}min` : 'N/A';
            const resolution = run.resolution || 'N/A';

            let badges = '';
            if (run.event_counts) {
                if (run.event_counts.shot > 0) {
                    badges += `<span class="event-badge shot">${run.event_counts.shot} shots</span>`;
                }
                if (run.event_counts.goal > 0) {
                    badges += `<span class="event-badge goal">${run.event_counts.goal} goals</span>`;
                }
            }

            html += `
                <div class="run-item" onclick="loadRun('${run.name}')">
                    <div class="run-name">${run.name}</div>
                    <div class="run-meta">
                        ${duration} • ${resolution} • ${run.fps ? run.fps.toFixed(0) + 'fps' : 'N/A'}
                        <div style="margin-top: 0.5rem;">${badges}</div>
                    </div>
                </div>
            `;
        });

        runsList.innerHTML = html;
    } catch (error) {
        console.error('Error loading runs:', error);
        runsList.innerHTML = '<h2>Analysis Runs</h2><p class="loading">Error loading runs</p>';
    }
}

// Load specific run
async function loadRun(runName) {
    currentRun = runName;

    // Update UI
    document.querySelectorAll('.run-item').forEach(item => item.classList.remove('active'));
    event.target.closest('.run-item')?.classList.add('active');

    viewer.classList.add('active');

    // Load video
    videoSource.src = `/api/runs/${runName}/video`;
    videoPlayer.load();

    // Load events
    await loadEvents(runName);

    // Load timeline
    await loadTimeline(runName);
}

// Load events for a run
async function loadEvents(runName) {
    try {
        const response = await fetch(`/api/runs/${runName}/events`);
        const data = await response.json();
        events = data.events;

        renderEvents();
        renderTimelineMarkers();
    } catch (error) {
        console.error('Error loading events:', error);
        eventsList.innerHTML = '<p class="loading">No events found</p>';
    }
}

// Load score timeline
async function loadTimeline(runName) {
    try {
        const response = await fetch(`/api/runs/${runName}/timeline`);
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
    }
}

// Render events list
function renderEvents() {
    if (events.length === 0) {
        eventsList.innerHTML = '<p class="loading">No events detected</p>';
        return;
    }

    let html = '';
    events.forEach((event, index) => {
        const time = formatTime(event.timestamp);
        const confidence = event.confidence.toFixed(2);
        const confidenceClass = event.confidence >= 0.7 ? 'high' : '';

        let details = '';
        if (event.metadata) {
            if (event.event_type === 'shot' && event.metadata.speed) {
                details += `Speed: ${event.metadata.speed.toFixed(1)}px/f`;
            }
            if (event.metadata.target_goal) {
                details += ` • Target: ${event.metadata.target_goal}`;
            }
            if (event.metadata.goal_region) {
                details += ` • Region: ${event.metadata.goal_region}`;
            }
        }

        html += `
            <div class="event-item ${event.event_type}" onclick="seekToEvent(${event.timestamp})">
                <div class="event-time">
                    ${event.event_type.toUpperCase()} at ${time}
                    <span class="confidence ${confidenceClass}">${confidence}</span>
                </div>
                <div class="event-details">${details}</div>
            </div>
        `;
    });

    eventsList.innerHTML = html;
}

// Render timeline markers
function renderTimelineMarkers() {
    const duration = videoPlayer.duration || 1;

    // Clear existing markers
    const existingMarkers = timelineBar.querySelectorAll('.timeline-marker');
    existingMarkers.forEach(marker => marker.remove());

    events.forEach(event => {
        const marker = document.createElement('div');
        marker.className = `timeline-marker ${event.event_type}`;
        marker.style.left = `${(event.timestamp / duration) * 100}%`;
        marker.title = `${event.event_type} at ${formatTime(event.timestamp)}`;
        marker.onclick = (e) => {
            e.stopPropagation();
            seekToEvent(event.timestamp);
        };
        timelineBar.appendChild(marker);
    });
}

// Setup video player
function setupVideoPlayer() {
    // Update progress bar
    videoPlayer.addEventListener('timeupdate', () => {
        if (videoPlayer.duration) {
            const progress = (videoPlayer.currentTime / videoPlayer.duration) * 100;
            timelineProgress.style.width = `${progress}%`;
        }
    });

    // Add markers when video loads
    videoPlayer.addEventListener('loadedmetadata', () => {
        renderTimelineMarkers();
    });

    // Click timeline to seek
    timelineBar.addEventListener('click', (e) => {
        const rect = timelineBar.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percentage = x / rect.width;
        videoPlayer.currentTime = percentage * videoPlayer.duration;
    });
}

// Seek to event time
function seekToEvent(timestamp) {
    videoPlayer.currentTime = timestamp;
    videoPlayer.play();
}

// Format time as MM:SS
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}
