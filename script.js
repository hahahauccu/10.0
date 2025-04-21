const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const poseImage = document.getElementById('poseImage');
const progressBar = document.getElementById('progressBar');
const restartBtn = document.getElementById('restartBtn');

let detector, rafId;
let currentPoseIndex = 0;
const totalPoses = 7;
let standardKeypointsList = [];
let poseOrder = [];

let holdStartTime = null;
const holdDuration = 3000;

// ✅ 隨機順序
function shufflePoseOrder() {
  poseOrder = Array.from({ length: totalPoses }, (_, i) => i + 1);
  for (let i = poseOrder.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [poseOrder[i], poseOrder[j]] = [poseOrder[j], poseOrder[i]];
  }
}

// ✅ 嘗試載入 png 或 PNG
function resolvePoseImageName(base) {
  const png = `poses/${base}.png`;
  const PNG = `poses/${base}.PNG`;
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => resolve(png);
    img.onerror = () => resolve(PNG);
    img.src = png;
  });
}

// ✅ 載入所有 JSON 和圖
async function loadStandardKeypoints() {
  standardKeypointsList = [];
  for (const i of poseOrder) {
    const res = await fetch(`poses/pose${i}.json`);
    const json = await res.json();
    const keypoints = json.keypoints || json;
    standardKeypointsList.push({
      id: i,
      keypoints,
      imagePath: await resolvePoseImageName(`pose${i}`)
    });
  }
}

// ✅ 畫出骨架
function drawKeypoints(kps, color, radius, alpha) {
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  kps.forEach(kp => {
    if (kp.score > 0.4) {
      ctx.beginPath();
      ctx.arc(kp.x, kp.y, radius, 0, 2 * Math.PI);
      ctx.fill();
    }
  });
  ctx.globalAlpha = 1.0;
}

// ✅ 計算關節角度
function getAngle(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const magAB = Math.hypot(ab.x, ab.y);
  const magCB = Math.hypot(cb.x, cb.y);
  if (magAB * magCB === 0) return 180;
  const cos = dot / (magAB * magCB);
  return Math.acos(Math.max(-1, Math.min(1, cos))) * (180 / Math.PI);
}

// ✅ 角度比對
function compareKeypointsAngleBased(a, b) {
  const joints = [
    ["left_shoulder", "left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow", "right_wrist"],
    ["left_hip", "left_knee", "left_ankle"],
    ["right_hip", "right_knee", "right_ankle"],
    ["left_elbow", "left_shoulder", "left_hip"],
    ["right_elbow", "right_shoulder", "right_hip"]
  ];

  let totalDiff = 0, count = 0;
  for (const [p1, p2, p3] of joints) {
    const kpA = [a.find(p => p.name === p1), a.find(p => p.name === p2), a.find(p => p.name === p3)];
    const kpB = [b.find(p => p.name === p1), b.find(p => p.name === p2), b.find(p => p.name === p3)];
    if (kpA.every(p => p && p.score > 0.4) && kpB.every(p => p && p.score > 0.4)) {
      const angleA = getAngle(kpA[0], kpA[1], kpA[2]);
      const angleB = getAngle(kpB[0], kpB[1], kpB[2]);
      totalDiff += Math.abs(angleA - angleB);
      count++;
    }
  }

  if (!count) return 0;
  const avgDiff = totalDiff / count;
  return avgDiff < 45 ? 1 : 0; // ✅ 判定標準（越小越嚴格）
}

// ✅ 偵測流程
async function detect() {
  const result = await detector.estimatePoses(video);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const currentPose = standardKeypointsList[currentPoseIndex];
  if (currentPose) drawKeypoints(currentPose.keypoints, 'blue', 6, 0.5);

  if (result.length > 0) {
    const user = result[0].keypoints;
    drawKeypoints(user, 'red', 6, 1.0);

    const sim = compareKeypointsAngleBased(user, currentPose.keypoints);
    const now = performance.now();

    if (sim === 1) {
      if (!holdStartTime) holdStartTime = now;
      const heldTime = now - holdStartTime;
      const progressPercent = Math.min(100, (heldTime / holdDuration) * 100);
      progressBar.style.width = `${progressPercent}%`;

      if (heldTime >= holdDuration) {
        holdStartTime = null;
        progressBar.style.width = "0%";
        currentPoseIndex++;
        if (currentPoseIndex < totalPoses) {
          poseImage.src = standardKeypointsList[currentPoseIndex].imagePath;
        } else {
          cancelAnimationFrame(rafId);
          poseImage.src = "";
          restartBtn.style.display = "block";
        }
      }
    } else {
      holdStartTime = null;
      progressBar.style.width = "0%";
    }
  }

  rafId = requestAnimationFrame(detect);
}

// ✅ 啟動
async function startGame() {
  startBtn.disabled = true;
  startBtn.style.display = 'none';
  restartBtn.style.display = 'none';

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: { exact: 'environment' },
      width: { ideal: 640 },
      height: { ideal: 480 }
    },
    audio: false
  });
  video.srcObject = stream;
  await video.play();

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.setTransform(-1, 0, 0, 1, canvas.width, 0); // ✅ 鏡像

  // ✅ WebGL → WASM → CPU fallback
  try {
    await tf.setBackend('webgl');
    await tf.ready();
  } catch {
    try {
      await tf.setBackend('wasm');
      await tf.ready();
    } catch {
      await tf.setBackend('cpu');
      await tf.ready();
    }
  }

  detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING }
  );

  shufflePoseOrder();
  await loadStandardKeypoints();
  currentPoseIndex = 0;
  poseImage.src = standardKeypointsList[0].imagePath;
  detect();
}

startBtn.addEventListener("click", startGame);
restartBtn.addEventListener("click", startGame);

// ✅ 點一下畫面也能跳下一動作
document.body.addEventListener('click', () => {
  if (!standardKeypointsList.length) return;
  currentPoseIndex++;
  holdStartTime = null;
  progressBar.style.width = "0%";
  if (currentPoseIndex < totalPoses) {
    poseImage.src = standardKeypointsList[currentPoseIndex].imagePath;
  } else {
    cancelAnimationFrame(rafId);
    poseImage.src = "";
    restartBtn.style.display = "block";
  }
});
