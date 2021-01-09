// 1. Install dependencies DONE
// 2. Import dependencies DONE
// 3. Setup webcam and canvas DONE
// 4. Define references to those DONE
// 5. Load posenet DONE
// 6. Detect function DONE
// 7. Drawing utilities from tensorflow DONE
// 8. Draw functions DONE

import React, { useRef } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
import * as posenet from "@tensorflow-models/posenet";
import Webcam from "react-webcam";
import { drawKeypoints, drawSkeleton } from "./util";
import logo from './astrobots_logo.png';

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const videoWidth = 640;
  const videoHeight = 480;

  const detect = async (net) => {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      const ctx = canvasRef.current.getContext("2d");
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      async function poseFrame() {
        let poses = [];
        let minPoseConfidence;
        let minPartConfidence;

        const pose = await net.estimatePoses(video, {
          flipHorizontal: true,
          decodingMethod: 'multi-person',
          maxDetections: 5,
          scoreThreshold: 0.1,
          nmsRadius: 30.0
        });
        poses = pose.concat(pose);
        minPoseConfidence = +0.15;
        minPartConfidence = +0.1;

        ctx.clearRect(0, 0, videoWidth, videoHeight);
        ctx.save();
        ctx.scale(-1, 1);
        ctx.translate(-videoWidth, 0);
        ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
        ctx.restore();

        poses.forEach(({score, keypoints}) => {
          if (score >= 0.15) {
            drawKeypoints(keypoints, 0.1, ctx);
            drawSkeleton(keypoints, 0.1, ctx);
          }
        });

        requestAnimationFrame(poseFrame);
      }

      poseFrame();
  };

  //  Load posenet
  const runPosenet = async () => {
    const net = await posenet.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      inputResolution: {width: 257, height: 200},
      multiplier: 0.75
    });

    detect(net);
  };

  runPosenet();

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} alt="Logo" 
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            top: 30,
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 10,
          }}
        />

        <Webcam
          ref={webcamRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            top: 350,
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 640,
            height: 480,
          }}
        />

        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            top: 350,
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 640,
            height: 480,
          }}
        />
      </header>
    </div>
  );
}

export default App;
