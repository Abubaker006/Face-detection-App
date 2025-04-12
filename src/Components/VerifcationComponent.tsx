"use client";
import React, { useState, useRef, useEffect } from "react";
import styles from "./Verification.module.css";
import * as faceapi from "face-api.js";
import { getCenterPoint } from "face-api.js/build/commonjs/utils";

interface FaceData {
  descriptor: Float32Array;
  detection: faceapi.WithFaceDescriptor<{
    detection: faceapi.FaceDetection;
    landmarks: faceapi.FaceLandmarks68;
  }>;
}

interface VerificationResult {
  isMatch: boolean;
  message: string;
  distance?: number;
  confidence?: number;
}

const VerificationComponent = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [modelsLoaded, setModelsLoaded] = useState<boolean>(false);
  const [initialFace, setInitialFace] = useState<FaceData | null>(null);
  const [verificationResult, setVerificationResult] =
    useState<VerificationResult | null>(null);
  const [statusMessage, setStatusMessage] =
    useState<string>("Loading Models....");
  const [multipleFacesDetected, setMultipleFacesDetected] =
    useState<boolean>(false);
  const [isLookingAtScreen, setIsLookingAtScreen] = useState<boolean>(true);

  const MODEL_URL = "/models";

  const loadModels = async (): Promise<void> => {
    try {
      setStatusMessage("Loading face detection models");
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
      ]);

      setModelsLoaded(true);
      setStatusMessage("Models loaded successfully, you can now capture face.");
    } catch (error) {
      console.error("error loading the models", error);
      setStatusMessage("Failed to load models. Check console for details.");
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  const startWebcam = async (): Promise<void> => {
    if (!videoRef.current) return;
    try {
      setStatusMessage("Requesting webcam access");
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      videoRef.current.srcObject = stream;
      setStatusMessage(
        'Webcam active. Position your face and click "Capture Face"'
      );
    } catch (error) {
      console.error("Failed at accessing web cam", error);
      setStatusMessage(
        "Failed to access webcam. Check permissions and try again."
      );
    }
  };

  useEffect(() => {
    if (modelsLoaded) {
      startWebcam();
    }
  }, [modelsLoaded]);

  const checkIfLookingAtScreen = (
    landmarks: faceapi.FaceLandmarks68
  ): boolean => {
    const nose = landmarks.getNose();
    const leftEye = landmarks.getLeftEye();
    const rightEye = landmarks.getRightEye();

    const nosePoint = nose[3];
    const leftEyeCenter = getCenterPoint(leftEye);
    const rightEyeCenter = getCenterPoint(rightEye);

    const eyeLine = {
      x: rightEyeCenter.x - leftEyeCenter.x,
      y: rightEyeCenter.y - leftEyeCenter.y,
    };

    const eyeYDifference = Math.abs(eyeLine.y);
    const eyeXDifference = Math.abs(eyeLine.x);

    const noseXCenter = (leftEyeCenter.x + rightEyeCenter.x) / 2;
    const noseOffset = Math.abs(nosePoint.x - noseXCenter);

    const isEyesLevel = eyeYDifference < eyeXDifference * 0.3;
    const isNoseCentered = noseOffset < eyeXDifference * 0.4;

    return isEyesLevel && isNoseCentered;
  };

  const captureFace = async () => {
    if (!videoRef.current || !modelsLoaded) return;
    try {
      setStatusMessage("Detecting Face....");
      const detections = await faceapi
        .detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptors();

      if (detections.length === 0) {
        setStatusMessage(
          "No face detected! Please position yourself properly and try again"
        );
        return;
      }

      if (detections.length > 1) {
        setMultipleFacesDetected(true);
        setStatusMessage(
          "Multiple faces detected! Please ensure only your face is in the frame."
        );

        if (canvasRef.current) {
          const displaySize = {
            width: videoRef.current.width,
            height: videoRef.current.height,
          };
          faceapi.matchDimensions(canvasRef.current, displaySize);
          const resizedDetections = faceapi.resizeResults(
            detections,
            displaySize
          );

          const ctx = canvasRef.current.getContext("2d");
          if (!ctx) {
            setStatusMessage("Canvas not initialized");
            return;
          }
          ctx.clearRect(
            0,
            0,
            canvasRef.current.width,
            canvasRef.current.height
          );
          faceapi.draw.drawDetections(canvasRef.current, resizedDetections);
          faceapi.draw.drawFaceLandmarks(canvasRef.current, resizedDetections);

          ctx.font = "24px Arial";
          ctx.fillStyle = "red";
          ctx.fillText("MULTIPLE FACES DETECTED", 10, 30);
        }
        return;
      }

      const detection = detections[0];

      const isLooking = checkIfLookingAtScreen(detection.landmarks);
      setIsLookingAtScreen(isLooking);

      if (!isLooking) {
        setStatusMessage(
          "Please look directly at the screen before capturing your face."
        );

        if (canvasRef.current) {
          const displaySize = {
            width: videoRef.current.width,
            height: videoRef.current.height,
          };
          faceapi.matchDimensions(canvasRef.current, displaySize);
          const resizedDetection = faceapi.resizeResults(
            detection,
            displaySize
          );

          const ctx = canvasRef.current.getContext("2d");
          if (!ctx) {
            setStatusMessage("Canvas not initialized");
            return;
          }
          ctx.clearRect(
            0,
            0,
            canvasRef.current.width,
            canvasRef.current.height
          );
          faceapi.draw.drawDetections(canvasRef.current, [resizedDetection]);
          faceapi.draw.drawFaceLandmarks(canvasRef.current, [resizedDetection]);

          // Draw warning text
          ctx.font = "24px Arial";
          ctx.fillStyle = "orange";
          ctx.fillText("PLEASE LOOK AT SCREEN", 10, 30);
        }
        return;
      }

      console.log("Detected data", detection);
      setInitialFace({
        descriptor: detection.descriptor,
        detection: detection,
      });

      if (canvasRef.current) {
        const displaySize = {
          width: videoRef.current.width,
          height: videoRef.current.height,
        };
        faceapi.matchDimensions(canvasRef.current, displaySize);
        const resizedDetection = faceapi.resizeResults(detection, displaySize);
        const ctx = canvasRef.current.getContext("2d");
        if (!ctx) {
          setStatusMessage("Canvas not initialized");
          return;
        }
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        faceapi.draw.drawDetections(canvasRef.current, [resizedDetection]);
        faceapi.draw.drawFaceLandmarks(canvasRef.current, [resizedDetection]);

        ctx.font = "24px Arial";
        ctx.fillStyle = "green";
        ctx.fillText("FACE CAPTURED ✓", 10, 30);
      }

      setStatusMessage("Face captured successfully! Now try the verification.");
    } catch (error) {
      console.error("Error at capturing initial snap of user", error);
      setStatusMessage("Failed at detecting face.");
    }
  };

  const verifyFace = async () => {
    console.log("Modal", modelsLoaded);
    console.log("initial Face", initialFace);
    console.log("video ref", videoRef.current);

    if (!videoRef.current || !initialFace || !modelsLoaded) {
      setStatusMessage("Please capture your initial face first");
      return;
    }

    try {
      setStatusMessage("Verifying face");

      const detections = await faceapi
        .detectAllFaces(videoRef.current, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptors();

      if (detections.length === 0) {
        setVerificationResult({
          isMatch: false,
          message: "No face detected for verification.",
        });
        setStatusMessage("Verification failed: No face detected.");
        return;
      }

      if (detections.length > 1) {
        setMultipleFacesDetected(true);
        setVerificationResult({
          isMatch: false,
          message:
            "Multiple faces detected. Please ensure only your face is in the frame.",
        });

        if (canvasRef.current) {
          const displaySize = {
            width: videoRef.current.width,
            height: videoRef.current.height,
          };
          faceapi.matchDimensions(canvasRef.current, displaySize);
          const resizedDetections = faceapi.resizeResults(
            detections,
            displaySize
          );

          const ctx = canvasRef.current.getContext("2d");
          if (!ctx) {
            setStatusMessage("Canvas not initialized");
            return;
          }
          ctx.clearRect(
            0,
            0,
            canvasRef.current.width,
            canvasRef.current.height
          );
          faceapi.draw.drawDetections(canvasRef.current, resizedDetections);
          faceapi.draw.drawFaceLandmarks(canvasRef.current, resizedDetections);

          ctx.font = "24px Arial";
          ctx.fillStyle = "red";
          ctx.fillText("MULTIPLE FACES DETECTED", 10, 30);
        }

        setStatusMessage("Verification failed: Multiple faces detected.");
        return;
      }

      const detection = detections[0];

      const isLooking = checkIfLookingAtScreen(detection.landmarks);
      setIsLookingAtScreen(isLooking);

      if (!isLooking) {
        setVerificationResult({
          isMatch: false,
          message: "Please look directly at the screen for verification.",
        });

        if (canvasRef.current) {
          const displaySize = {
            width: videoRef.current.width,
            height: videoRef.current.height,
          };
          faceapi.matchDimensions(canvasRef.current, displaySize);
          const resizedDetection = faceapi.resizeResults(
            detection,
            displaySize
          );

          const ctx = canvasRef.current.getContext("2d");
          if (!ctx) {
            setStatusMessage("Canvas not initialized");
            return;
          }
          ctx.clearRect(
            0,
            0,
            canvasRef.current.width,
            canvasRef.current.height
          );
          faceapi.draw.drawDetections(canvasRef.current, [resizedDetection]);
          faceapi.draw.drawFaceLandmarks(canvasRef.current, [resizedDetection]);

          ctx.font = "24px Arial";
          ctx.fillStyle = "orange";
          ctx.fillText("PLEASE LOOK AT SCREEN", 10, 30);
        }

        setStatusMessage("Verification failed: Not looking at screen.");
        return;
      }

      const distance = faceapi.euclideanDistance(
        initialFace.descriptor,
        detection.descriptor
      );

      const threshold = 0.6;
      const isMatch = distance < threshold;
      const confidence = Math.round((1 - distance) * 100);

      setVerificationResult({
        isMatch,
        distance,
        confidence,
        message: isMatch
          ? `Match confirmed! (${confidence}% confidence)`
          : `Different person detected. (${confidence}% similarity)`,
      });

      if (canvasRef.current) {
        const displaySize = {
          width: videoRef.current.width,
          height: videoRef.current.height,
        };
        faceapi.matchDimensions(canvasRef.current, displaySize);
        const resizedDetection = faceapi.resizeResults(detection, displaySize);

        const ctx = canvasRef.current.getContext("2d");
        if (!ctx) {
          setStatusMessage("Canvas not initialized");
          return;
        }
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        faceapi.draw.drawDetections(canvasRef.current, [resizedDetection]);
        faceapi.draw.drawFaceLandmarks(canvasRef.current, [resizedDetection]);

        ctx.font = "24px Arial";
        ctx.fillStyle = isMatch ? "green" : "red";
        ctx.fillText(isMatch ? "MATCH ✓" : "NO MATCH ✗", 10, 30);
      }

      setStatusMessage(
        `Verification complete: ${
          isMatch ? "Match confirmed!" : "Different person detected."
        }`
      );
    } catch (error) {
      console.error("Error at verifying face", error);
      setStatusMessage("Error at verifying the face.");
    }
  };

  return (
    <div className={styles.container}>
      <h1>Face Verification Test Project</h1>

      <p className={styles.status}>{statusMessage}</p>

      {multipleFacesDetected && (
        <div className={styles.warning}>
          <p>
            Multiple faces detected! For proper verification, ensure only your
            face is visible in the camera.
          </p>
        </div>
      )}

      {!isLookingAtScreen && (
        <div className={styles.warning}>
          <p>
            Please look directly at the screen for proper face verification.
          </p>
        </div>
      )}

      <div className={styles.videoContainer}>
        <video
          ref={videoRef}
          width="640"
          height="480"
          autoPlay
          muted
          className={styles.video}
        />
        <canvas
          ref={canvasRef}
          width="640"
          height="480"
          className={styles.canvas}
        />
      </div>

      <div className={styles.controls}>
        <button
          onClick={captureFace}
          disabled={!modelsLoaded}
          className={styles.button}
        >
          Capture Initial Face
        </button>

        <button
          onClick={verifyFace}
          disabled={!initialFace}
          className={styles.button}
        >
          Verify Current Face
        </button>
      </div>

      {verificationResult && (
        <div
          className={`${styles.result} ${
            verificationResult.isMatch ? styles.match : styles.noMatch
          }`}
        >
          <h2>Verification Result:</h2>
          <p>{verificationResult.message}</p>
          {verificationResult.distance && (
            <p>
              Distance score: {verificationResult.distance.toFixed(2)}
              (Lower is more similar. Threshold: 0.6)
            </p>
          )}
        </div>
      )}

      <div className={styles.instructions}>
        <h3>Instructions:</h3>
        <ol>
          <li>Wait for models to load and webcam to activate</li>
          <li>
            Position your face in the frame and click &quot;Capture Initial
            Face&quot;
          </li>
          <li>
            For verification test, either stay in frame (should match) or have
            someone else step in (should not match)
          </li>
          <li>
            Click &quot;Verify Current Face&quot; to check if the current face
            matches the initial one
          </li>
        </ol>
      </div>
    </div>
  );
};

export default VerificationComponent;
