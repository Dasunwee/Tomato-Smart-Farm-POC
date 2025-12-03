import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const SmartFarmApp());
}

class SmartFarmApp extends StatelessWidget {
  const SmartFarmApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Tomato Smart Farm',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.green),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}


class _HomePageState extends State<HomePage> {
  // --- 1. CONFIGURATION ---
  // Your live Azure API URL
  final String apiUrl = "https://tomato-farm-app-dasun.azurewebsites.net/analyze";
  
  File? _selectedImage;
  bool _isLoading = false;
  Map<String, dynamic>? _apiResponse;

  // --- 2. IMAGE PICKER LOGIC ---
  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source);

    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
        _apiResponse = null; // Reset previous results
      });
    }
  }

  // --- 3. API UPLOAD LOGIC ---
  Future<void> _analyzePlant() async {
    if (_selectedImage == null) return;

    setState(() {
      _isLoading = true;
    });

    try {
      // Create a Multipart Request (like a form upload)
      var request = http.MultipartRequest('POST', Uri.parse(apiUrl));
      request.files.add(await http.MultipartFile.fromPath('file', _selectedImage!.path));

      // Send Request
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        setState(() {
          _apiResponse = jsonDecode(response.body);
        });
      } else {
        _showError("Server Error: ${response.statusCode}");
      }
    } catch (e) {
      _showError("Connection Error: $e");
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
  }

  // --- 4. UI BUILDING ---
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('🍅 Smart Farm AI'),
        backgroundColor: Colors.green.shade100,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // --- Image Display ---
            Container(
              height: 250,
              decoration: BoxDecoration(
                color: Colors.grey.shade200,
                borderRadius: BorderRadius.circular(12),
                image: _selectedImage != null
                    ? DecorationImage(image: FileImage(_selectedImage!), fit: BoxFit.cover)
                    : null,
              ),
              child: _selectedImage == null
                  ? const Center(child: Icon(Icons.camera_alt, size: 50, color: Colors.grey))
                  : null,
            ),
            const SizedBox(height: 16),

            // --- Action Buttons ---
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: () => _pickImage(ImageSource.camera),
                  icon: const Icon(Icons.camera),
                  label: const Text("Camera"),
                ),
                ElevatedButton.icon(
                  onPressed: () => _pickImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo),
                  label: const Text("Gallery"),
                ),
              ],
            ),
            const SizedBox(height: 16),

            // --- Analyze Button ---
            if (_selectedImage != null && !_isLoading)
              ElevatedButton(
                onPressed: _analyzePlant,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
                child: const Text("ANALYZE PLANT", style: TextStyle(fontSize: 18)),
              ),

            if (_isLoading) const Center(child: CircularProgressIndicator()),

            const SizedBox(height: 20),

            // --- Results Display ---
            if (_apiResponse != null) ...[
              const Text("Analysis Results", style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
              const SizedBox(height: 10),
              
              // Health Card
              _buildResultCard(
                title: "Plant Health",
                icon: Icons.health_and_safety,
                color: _apiResponse!['plant_health']['is_healthy'] ? Colors.green : Colors.red,
                content: [
                  "Status: ${_apiResponse!['plant_health']['status']}",
                  "Confidence: ${_apiResponse!['plant_health']['confidence']}",
                ],
              ),
              
              // Harvest Card
              _buildResultCard(
                title: "Harvest Info",
                icon: Icons.shopping_basket,
                color: Colors.orange,
                content: [
                  "Ready to Harvest: ${_apiResponse!['harvest_info']['ready_to_harvest'] ? 'YES' : 'NO'}",
                  "Ripe Fruits: ${_apiResponse!['harvest_info']['ripe_fruits']}",
                  "Unripe Fruits: ${_apiResponse!['harvest_info']['unripe_fruits']}",
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildResultCard({
    required String title,
    required IconData icon,
    required Color color,
    required List<String> content,
  }) {
    return Card(
      elevation: 4,
      margin: const EdgeInsets.only(bottom: 16),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(children: [
              Icon(icon, color: color, size: 28),
              const SizedBox(width: 10),
              Text(title, style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: color)),
            ]),
            const Divider(),
            ...content.map((text) => Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Text(text, style: const TextStyle(fontSize: 16)),
                )),
          ],
        ),
      ),
    );
  }
}