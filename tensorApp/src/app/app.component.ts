import { Component, OnInit, ViewChild } from '@angular/core';
import * as tf from 'node_modules/@tensorflow/tfjs';
import { DrawableDirective } from './drawable.directive';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
})
export class AppComponent implements OnInit {
  title = 'tensorApp';

  linearModel: tf.Sequential;
  prediction: any;

  model: tf.LayersModel;
  @ViewChild(DrawableDirective) canvas;
  ngOnInit(): void {
    this.trainNewModel();
    this.loadModel();
  }

  async loadModel() {
    var x = tf.loadLayersModel('/assets/model.json')
    //this.model = await tf.loadModel('/assets/model.json');
  }

  // async predict(imageData: ImageData) {
  //   const pred = await tf.tidy(() => {

  //     // Convert the canvas pixels to 
  //     let img = tf.fromPixels(imageData, 1);
  //     img = img.reshape([1, 28, 28, 1]);
  //     img = tf.cast(img, 'float32');

  //     // Make and format the predications
  //     const output = this.model.predict(img) as any;

  //     // Save predictions on the component
  //     this.predictions = Array.from(output.dataSync()); 
  //   });

  // }

  async trainNewModel():Promise<any>{
    this.linearModel = tf.sequential();
    this.linearModel.add(tf.layers.dense({units: 1, inputShape: [1]}));

    this.linearModel.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const xs = tf.tensor1d([3.2, 4.4, 5.5, 6.71, 6.98, 7.168, 9.779, 6.182, 7.59, 2.16, 7.0]);
    const ys = tf.tensor1d([1.6, 2.7, 2.9, 3.19, 1.684, 2.53, 3.366, 2.596, 2.53, 1.22, 2.8]);

    await this.linearModel.fit(xs, ys)

  }
  
  linearPredict(val: any) {
    var value = Number(val)
    const output = this.linearModel.predict(tf.tensor2d([value], [1, 1])) as any
    console.log(output)
    this.prediction = Array.from(output.dataSync())[0]
  }

}
