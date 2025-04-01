import 'package:flutter/services.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

// TODO: add pooling

Vector reduce(Matrix m){
    var vec = m.reduceRows((v,w)=>v+w, initValue: Vector.zero(m.columnCount));
    vec/=m.rowCount;
    vec/=vec.norm();
    return vec;
}

class EmbeddingModel{
  OrtSession? model;
  EmbeddingModel(this.model);

  static Future<EmbeddingModel> fromAssetFile(String path) async{
    final rawAssetFile = await rootBundle.load(path);
    final bytes = rawAssetFile.buffer.asUint8List();

    final options = OrtSessionOptions();
    final session = OrtSession.fromBuffer(bytes, options);

    return EmbeddingModel(session);
  }

  Future<Vector> forward(Int64List inputIds, Int64List attentionMask, Int64List tokenTypes) async{
    final inputIdsTensor = OrtValueTensor.createTensorWithDataList(inputIds.inner.toList(), [1, inputIds.length]);
    final attentionMaskTensor = OrtValueTensor.createTensorWithDataList(attentionMask.inner.toList(), [1, attentionMask.length]);
    final tokenTypesTensor = OrtValueTensor.createTensorWithDataList(tokenTypes.inner.toList(), [1, tokenTypes.length]);

    final inputs = {
      'input_ids': inputIdsTensor, 
      'attention_mask': attentionMaskTensor, 
      'token_type_ids': tokenTypesTensor, 
    };

    final runOptions = OrtRunOptions();
    final outputs = await model?.runAsync(runOptions, inputs);

    // mean pool hidden states
    final preHiddenStates = outputs![0]!.value! as List<List<List<double>>>;
    final hiddenStates = Matrix.fromList(preHiddenStates[0]);
    final embeddings = reduce(hiddenStates);

    // cleanup 
    runOptions.release();
    outputs?.forEach((element) {
      element?.release();
    });

    inputIdsTensor.release();
    attentionMaskTensor.release();
    tokenTypesTensor.release();
    
    // final hiddenStatesMatrix = Matrix.fromByteData();
    return embeddings;
  }
}
