package com.example.comparing;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;


import com.google.android.gms.tasks.OnSuccessListener;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    protected Interpreter tflite;
    private int imageSizeX;
    private int imageSizeY;
    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD = 1.0f;

    public Bitmap originalImage,testImage;
    public static Bitmap cropped;
    Uri imageUri;

    ImageView original,test;
    Button view;
    TextView result;


    float[][] originalEmbedding = new float[1][128];
    float[][] testEmbedding = new float[1][128];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initComponents();
    }

    private void initComponents(){
        original = findViewById(R.id.original);
        test = findViewById(R.id.test);
        view = findViewById(R.id.view);
        result = findViewById(R.id.resultText);
        try {
            tflite = new Interpreter(loadModelFile(this));
        }catch(Exception e){
            e.printStackTrace();
        }
        original.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent,"Select Picture"),12);
            }
        });
        test.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent,"Select Picture"),13);
            }
        });
        view.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                double distance = calculateDistance(originalEmbedding,testEmbedding);
                if (distance<6.0){
                    result.setText("Result: same faces");
                }else {
                    result.setText("Result: not same faces");
                }
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == 12 && resultCode==RESULT_OK && data!=null) {
            imageUri = data.getData();
            try {
                originalImage = MediaStore.Images.Media.getBitmap(getContentResolver(),imageUri);
                original.setImageBitmap(originalImage);
                faceDetector(originalImage,"original");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        if (requestCode == 13 && resultCode==RESULT_OK && data!=null) {
            imageUri = data.getData();
            try {
                testImage = MediaStore.Images.Media.getBitmap(getContentResolver(),imageUri);
                test.setImageBitmap(testImage);
                faceDetector(testImage,"test");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    public void faceDetector(final Bitmap bitmap,final String imageType){
        final InputImage image = InputImage.fromBitmap(bitmap,0);
        FaceDetector detector = FaceDetection.getClient();
        detector.process(image)
                .addOnSuccessListener(
                        (OnSuccessListener<List<Face>>) (faces)->{
                            for(Face face : faces){
                                Rect bounds = face.getBoundingBox();
                                cropped = Bitmap.createBitmap(bitmap,bounds.left,bounds.top,bounds.width(),bounds.height());
                                getEmbeddings(cropped,imageType);
                            }
                        })
                .addOnFailureListener(
                        (e)->{
                            Toast.makeText(getApplicationContext(), e.getMessage(), Toast.LENGTH_SHORT).show();
                        }
                );
        }
    public void getEmbeddings(Bitmap bitmap,String imageType){
        TensorImage inputImageBuffer;
        float[][] embedding = new float[1][128];

        int imageTensorIndex = 0;
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape();
        imageSizeX = imageShape[1];
        imageSizeY = imageShape[2];
        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();

        inputImageBuffer = new TensorImage(imageDataType);

        inputImageBuffer = loadImage(bitmap,inputImageBuffer);

        tflite.run(inputImageBuffer.getBuffer(),embedding);

        if(imageType.equals("original")){
            originalEmbedding = embedding;
        } else if (imageType.equals("test")) {
            testEmbedding = embedding;
        }
    }
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException{
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd("Qfacenet.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffSet = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffSet,declaredLength);
    }
    private TensorImage loadImage(final Bitmap bitmap, TensorImage inputImageBuffer){
        inputImageBuffer.load(bitmap);

        int cropSize = Math.min(bitmap.getWidth(),bitmap.getHeight());
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeWithCropOrPadOp(cropSize,cropSize))
                .add(new ResizeOp(imageSizeX,imageSizeY,ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .add(getPreprocessNormalizeOp())
                .build();
        return imageProcessor.process(inputImageBuffer);
    }
    private TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }
    private double calculateDistance(float[][] originalEmbedding, float[][] testEmbedding){
        double sum = 0.0;
        for(int i =0;i<128;i++){
            sum += Math.pow((originalEmbedding[0][i]-testEmbedding[0][i]),2.0);
        }
        return Math.sqrt(sum);
    }
}