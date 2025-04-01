import 'package:flutter/material.dart';
import 'package:onnxruntime_example/model.dart';
import 'package:onnxruntime_example/src/rust/api/tokenize.dart';
import 'package:onnxruntime_example/src/rust/frb_generated.dart';

Future<void> main() async {
  await RustLib.init();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          leading: Icon(Icons.apple_outlined),
          title: const Text('onnxruntime & rust attempt'),
        ),
        body: Wrapper(),
      ),
    );
  }
}

class Wrapper extends StatefulWidget{
  @override
  State<Wrapper> createState() => _WrapperState();
}

class _WrapperState extends State<Wrapper> {
  late final EmbeddingModel model;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: FutureBuilder(future: loadModel(), builder: (context, res){
          if (res.hasData) {
            model = res.data!;
            return HomePage(model);
          }else{
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator()
                ],
              ),
            );
          }
      }),
    );
  }
}

class HomePage extends StatefulWidget {
  final EmbeddingModel model;
  const HomePage(this.model, {super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

Future<EmbeddingModel> loadModel() async =>
    await EmbeddingModel.fromAssetFile("assets/model.onnx");

class _HomePageState extends State<HomePage> {
  final controller1 = TextEditingController();
  final controller2 = TextEditingController();
  double? similarity; 
  late EmbeddingModel model;

  @override
    void initState() {
      super.initState();
      model = widget.model;
    }

  Future<void> computeSimilarity() async {
    final tokenizer = TokenizerWrapper.fromPretrained(modelId: "intfloat/e5-small-v2");
    final input1 = tokenizer.tokenize(prompt: controller1.text);
    final input2 = tokenizer.tokenize(prompt: controller2.text);

    final embedding1 = await model.forward(
      input1.inputIds,
      input1.attentionMask,
      input1.tokenTypeIds,
    );

    final embedding2 = await model.forward(
      input2.inputIds,
      input2.attentionMask,
      input2.tokenTypeIds,
    );

    setState(() {
      similarity = embedding1.dot(embedding2);
    });
  }

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 10.0),
              child: TextFormField(
                controller: controller1,
                decoration: InputDecoration(
                  border: OutlineInputBorder(),
                  labelText: 'First text input',
                  isDense: true,
                ),
              ),
            ),
          ),
          Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 10.0),
              child: TextFormField(
                controller: controller2,
                decoration: InputDecoration(
                  border: OutlineInputBorder(),
                  labelText: 'Second text input',
                  isDense: true,
                ),
              ),
            ),
          ),
          SizedBox(height: 10),
          FloatingActionButton(
            onPressed: computeSimilarity,
            child: Text("embed"),
          ),
          if (similarity != null) ...[
            SizedBox(height: 20),
            Text("Cosine Similarity: ${similarity!.toStringAsFixed(4)}"),
          ]
        ],
      ),
    );
  }

  @override
  void dispose() {
    controller1.dispose();
    controller2.dispose();
    super.dispose();
  }
}
