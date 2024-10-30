import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

// Entry point
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  final firstCamera = cameras.first;

  runApp(ChildSafetyApp(camera: firstCamera));
}

class ChildSafetyApp extends StatelessWidget {
  final CameraDescription camera;

  ChildSafetyApp({required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData.dark(),
      home: LiveFeedScreen(camera: camera),
    );
  }
}

class LiveFeedScreen extends StatefulWidget {
  final CameraDescription camera;

  LiveFeedScreen({required this.camera});

  @override
  _LiveFeedScreenState createState() => _LiveFeedScreenState();
}

class _LiveFeedScreenState extends State<LiveFeedScreen> {
  late CameraController _controller;
  late Future<void> _initializeControllerFuture;

  @override
  void initState() {
    super.initState();
    _controller = CameraController(widget.camera, ResolutionPreset.high);
    _initializeControllerFuture = _controller.initialize();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _detectRisk() async {
    try {
      final image = await _controller.takePicture();

      // Assume model endpoint URL (replace with your model API)
      final apiUrl = 'http://127.0.0.1:5000/predict';
      final response = await http.post(Uri.parse(apiUrl), body: {
        'image_path': image.path,
        // other data inputs for prediction can be added here
      });

      final result = response.body;
      // handle result (e.g., show alert if risky)
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: Text("Detection Result"),
          content: Text("Child status: $result"),
        ),
      );
    } catch (e) {
      print(e);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Child Safety Detection')),
      body: FutureBuilder<void>(
        future: _initializeControllerFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.done) {
            return Stack(
              children: [
                CameraPreview(_controller),
                Positioned(
                  bottom: 20,
                  left: 20,
                  child: ElevatedButton(
                    onPressed: _detectRisk,
                    child: Text('Check Safety'),
                  ),
                ),
              ],
            );
          } else {
            return Center(child: CircularProgressIndicator());
          }
        },
      ),
    );
  }
}
