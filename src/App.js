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

  const detect = async (net) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
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

        const pose = await net.estimatePoses(video, {
          flipHorizontal: true,
          decodingMethod: 'single-person'
        });
        poses = pose.concat(pose);

        ctx.clearRect(0, 0, videoWidth, videoHeight);
        ctx.save();
        ctx.scale(-1, 1);
        ctx.translate(-videoWidth, 0);
        ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
        ctx.restore();

        poses.forEach(({score, keypoints}) => {
          drawKeypoints(keypoints, 0.6, ctx);
          drawSkeleton(keypoints, 0.7, ctx);
        });

        requestAnimationFrame(poseFrame);
      }

      poseFrame();
    }
  };

  //  Load posenet
  const runPosenet = async () => {
    const net = await posenet.load({
      architecture: 'MobileNetV1',
      outputStride: 16,
      inputResolution: { width: 480, height: 400 },
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
            width: 480,
            height: 400,
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
            width: 480,
            height: 400,
          }}
        />
      </header>
    </div>
  );
}

export default App;
