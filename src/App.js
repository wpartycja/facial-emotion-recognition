import React, {useRef, useEffect, useState} from "react"
// import axios from 'axios';

function App() {

	const videoRef = useRef(null);
	const photoRef = useRef(null);
	
	const [hasPhoto, setHasPhoto] = useState(false);

	const getVideo = () => {
		navigator.mediaDevices
			.getUserMedia({
				video: {width: 1920, height: 1080}
			})
			.then(stream => {
				let video = videoRef.current;
				video.srcObject = stream;
				video.play();
			})
			.catch(err => {
				console.error(err);
			})
	}

	const takePhoto = async () => {
		const width = 414;
		const height = width / (16/9);

		let video = videoRef.current;
		let photo = photoRef.current;

		// sending the image

		let send_dicky = {
			name : 'Ola',
			surname : 'Jaj'
		}

		console.log(JSON.stringify(send_dicky))

		// fetch(
		// 	'http://localhost:8000/test',
		// 	{
		// 		method: 'POST',
		// 		body: JSON.stringify(send_dicky)
		// 		// , headers: {
		// 		// 	'Content-Type' : 'application/json'
		// 		// }
		// 	}
		// )

		const response = await fetch("http://localhost:8000/todo")
    	const todos = await response.json()
		console.log(todos)

		photo.width = width;
		photo.height = height;

		let ctx = photo.getContext('2d');
		ctx.drawImage(video, 0, 0, width, height);
		setHasPhoto(true);
	}



	const closePhoto = () => {
		let photo = photoRef.current;
		let ctx = photo.getContext('2d');
		ctx.clearRect(0,0,photo.width, photo.height)
		setHasPhoto(false);
	}

	useEffect(()=> {
		getVideo();
	}, [videoRef])

	return (
		<div className="App">
			<div className="camera"> 
				<video ref = {videoRef}></video>
				<button onClick={takePhoto}>SNAP!</button>
			</div>
			<div className={'result ' + (hasPhoto ? 'hasPhoto' : '')}>
				<canvas ref ={photoRef}></canvas>
				<button onClick={closePhoto}>CLOSE!</button>
			</div>
		</div>
	);
}

export default App;
