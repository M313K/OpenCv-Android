package com.example.opencvtest;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.app.assist.AssistStructure;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.util.Log;
import android.Manifest;
import android.widget.RelativeLayout;

import com.example.opencvtest.ml.Yolov5mBest2Fp162;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.CvType;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainActivity extends CameraActivity {

    CameraBridgeViewBase cameraBridgeViewBase;
    TensorBuffer outputFeature0;
    TensorBuffer inputFeature0;
    Yolov5mBest2Fp162 model;
    Yolov5mBest2Fp162.Outputs outputs;
    private Mat mRgba = null;
    private float mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY;
    //private Mat mGray = null;
    Mat mIntermediateMat = null;
    int imageSize = 640;
    Bitmap image;
    float[] outputArray;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getPermission();
        cameraBridgeViewBase = findViewById(R.id.cameraView);
       // cameraBridgeViewBase.setLayoutParams(new RelativeLayout.LayoutParams(640, 640));
        cameraBridgeViewBase.setCvCameraViewListener(new CameraBridgeViewBase.CvCameraViewListener2() {
            @Override
            public void onCameraViewStarted(int width, int height) {
                int mFrameWidth = width;
                int mFrameHeight = height;


                // Calculate scaling factors
                mImgScaleX = (float) cameraBridgeViewBase.getWidth() / width;
                mImgScaleY = (float) cameraBridgeViewBase.getHeight() / mFrameHeight;
                mIvScaleX = (float) mFrameWidth / cameraBridgeViewBase.getWidth();
                mIvScaleY = (float) mFrameHeight / cameraBridgeViewBase.getHeight();

                // Calculate starting point for ImageView
                mStartX = (cameraBridgeViewBase.getWidth() - mFrameWidth * mImgScaleX) / 2;
                mStartY = (cameraBridgeViewBase.getHeight() - mFrameHeight * mImgScaleY) / 2;
                mRgba = new Mat(height, width, CvType.CV_8UC4);
                //mGray = new Mat(height, width, CvType.CV_8UC4);
               // mIntermediateMat= new Mat(128, 128, CvType.CV_8UC4);
                //mRgba = new Mat(640, 640, CvType.CV_8UC4);


            }

            @Override
            public void onCameraViewStopped() {
                mRgba.release();
            }

            @Override
            public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
                mRgba = inputFrame.rgba();
                //convert Mat to Bitmap
                image = Bitmap.createBitmap(mRgba.cols(), mRgba.rows(), Bitmap.Config.ARGB_8888);

                Utils.matToBitmap(mRgba, image);
                // Creates inputs for reference.
                inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 640, 640, 3}, DataType.FLOAT32);
                ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
                byteBuffer.order(ByteOrder.nativeOrder());

                int[] intValues = new int[image.getWidth() * image.getHeight()];
                image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
                int pixel = 0;
                //iterate over each pixel and extract R, G, and B values. Add those values individually to the byte buffer.
                for(int i = 0; i < imageSize; i ++){
                    for(int j = 0; j < imageSize; j++){
                        int val = intValues[pixel++]; // RGB
                        byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255));
                        byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255));
                        byteBuffer.putFloat((val & 0xFF) * (1.f / 255));
                    }
                }

                inputFeature0.loadBuffer(byteBuffer);


                // Runs model inference and gets result.
                outputs = model.process(inputFeature0);
                TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
                //NMS
                outputArray = outputFeature0.getFloatArray();
               // Log.d("Output Array", String.valueOf(outputArray));
                final ArrayList<Result> results =  PrePostProcessor.outputsToNMSPredictions(outputArray, mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY,cameraBridgeViewBase.getHeight(),cameraBridgeViewBase.getWidth());
                Log.d("Output Array", String.valueOf(results));
                Log.d("Output scale", String.valueOf(mStartY)+String.valueOf(mStartX));
                // assume "results" is an ArrayList<Result> object containing detection results
                Scalar color = new Scalar(0, 255, 0); // set color to black (BGR format)
                int thickness = 3;
                for (Result result : results) {
                    int left = (int) result.rect.left;
                    int top = (int) result.rect.top;
                    int right = (int) result.rect.right;
                    int bottom = (int) result.rect.bottom;

// flip y-coordinates

                    Imgproc.rectangle(mRgba, new Point(left, top), new Point(right, bottom), color, thickness);
                }

// now every Result object in "results" has its own "rect" field containing the corresponding rectangle



                return mRgba;

            }
        });
        try {
            model = Yolov5mBest2Fp162.newInstance(this);



        } catch (IOException e) {
            // TODO Handle the exception
        }

        if(OpenCVLoader.initDebug())
            cameraBridgeViewBase.enableView();
        else
            Log.d("Loaded Opencv ","Error");


    }

    @Override
    protected void onResume() {
        super.onResume();
        cameraBridgeViewBase.enableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraBridgeViewBase.disableView();
        // Releases model resources if no longer used.
        model.close();

    }

    @Override
    protected void onPause() {
        super.onPause();
        cameraBridgeViewBase.disableView();
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(cameraBridgeViewBase);
    }

    void getPermission()
    {
        if(checkSelfPermission(Manifest.permission.CAMERA)!= PackageManager.PERMISSION_GRANTED)
        {
            requestPermissions(new String[]{Manifest.permission.CAMERA},101);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if(grantResults.length>0 && grantResults[0]!=PackageManager.PERMISSION_GRANTED )
        {
            getPermission();
        }
    }
}