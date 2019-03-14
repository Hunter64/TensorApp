import { Component, OnInit, ViewChild } from '@angular/core';
import * as tf from '@tensorflow/tfjs';


@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css'],
})
export class AppComponent implements OnInit {
  title = 'tensorApp';

  linearModel: tf.Sequential;
  prediction: any;

  //model: tf.model;
  //@ViewChild(DrawableDirective) canvas;
  ngOnInit(): void {
    this.trainNewModel();
    //this.loadModel();
  }

  async trainNewModel():Promise<any>{
    this.linearModel = tf.sequential();
    this.linearModel.add(tf.layers.dense({units: 1, inputShape: [1]}));

    this.linearModel.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    const xs = tf.tensor1d([3.2, 4.4, 5.5, 6.71, 6.98, 7.168, 9.779, 6.182, 7.59, 2.16, 7.0]);
    const ys = tf.tensor1d([1.6, 2.7, 2.9, 3.19, 1.684, 2.53, 3.366, 2.596, 2.53, 1.22, 2.8]);

    await this.linearModel.fit(xs, ys)

  }
  
  predict(val: any) {
    var value = Number(val)
    const output = this.linearModel.predict(tf.tensor2d([value], [1, 1])) as any
    console.log(output)
    this.prediction = Array.from(output.dataSync())[0]
  }

}
