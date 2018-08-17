'use strict';
const Canvas = require('canvas');
const fs = require('fs');
const path = require('path');
const seedrandom = require('seedrandom');
const jsonfile = require('jsonfile');

function drawPong(ctx, canvas, leftPaddleY, rightPaddleY, ballX, ballY) {
  const paddleThickness = 0.03 * canvas.width;
  const padding = paddleThickness;
  const paddleLength = 0.15 * canvas.height;
  const ballRadius = 0.03 * canvas.width;

  // background
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // paddles
  ctx.fillStyle = 'white';
  ctx.fillRect(
    padding,
    leftPaddleY * (canvas.height - paddleLength),
    paddleThickness,
    paddleLength
  );
  ctx.fillRect(
    canvas.width - padding - paddleThickness,
    rightPaddleY * (canvas.height - paddleLength),
    paddleThickness,
    paddleLength
  );

  // ball
  ctx.beginPath();
  ctx.arc(
    padding + paddleThickness + ballRadius + ballX * (canvas.width - 2 * padding - 2 * ballRadius - 2 * paddleThickness),
    ballRadius + ballY * (canvas.height - 2 * ballRadius),
    ballRadius,
    0,
    2 * Math.PI
  );
  ctx.fill();
}

const prng = seedrandom('deadbeef');

const numImages = 10000;
const splitIndex = 8000;
for (let i = 0; i < numImages; i++) {
  const leftPaddleY = prng();
  const rightPaddleY = prng();
  const ballX = prng();
  const ballY = prng();

  const fileName = `pong_${i}.png`;
  const subfolder = i < splitIndex ? 'training' : 'validation';
  const filePath = path.join(__dirname, 'data', subfolder, fileName);

  const metaData = {leftPaddleY, rightPaddleY, ballX, ballY};
  let metaDataFilePath = path.join(__dirname, 'data', subfolder, fileName + '.json');
  jsonfile.writeFileSync(metaDataFilePath, metaData);

  let canvas = new Canvas(256, 256);
  let ctx = canvas.getContext("2d");
  drawPong(ctx, canvas, leftPaddleY, rightPaddleY, ballX, ballY);

  let out = fs.createWriteStream(filePath);
  let stream = canvas.pngStream();

  stream.on('data', function(chunk) {
    out.write(chunk);
  });
  stream.on('end', function(){
    console.log(`${i}: Saved ${filePath}`);
  });
}
