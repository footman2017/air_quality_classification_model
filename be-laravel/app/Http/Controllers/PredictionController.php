<?php

namespace App\Http\Controllers;

use Carbon\Carbon;
use App\Models\Prediction;
use Illuminate\Http\Request;
use App\Events\NewDataInserted;
use Illuminate\Support\Facades\DB;

class PredictionController extends Controller
{
    //
    public function storePrediction(Request $request)
    {
        DB::beginTransaction();
        try {
            $prediction = new Prediction;
            $prediction->pm10 = $request->pm10;
            $prediction->pm25 = $request->pm25;
            $prediction->so2 = $request->so2;
            $prediction->co = $request->co;
            $prediction->o3 = $request->o3;
            $prediction->no2 = $request->no2;
            $prediction->prediction_result = $request->prediction_result;
            $prediction->accuracy = $request->accuracy;

            if ($prediction->save()) {
                DB::commit();
                event(new NewDataInserted($prediction->toArray()));
                return json_encode([
                    "status" => true,
                    "message" => "Success"
                ]);
            } else {
                DB::rollBack();
                return json_encode([
                    "status" => false,
                    "message" => "Error"
                ]);
            }
        } catch (\Throwable $th) {
            DB::rollBack();
            return json_encode([
                "status" => false,
                "message" => "Error",
                "data" => $th
            ]);
        }
    }

    public function getLatestPrediction()
    {
        $data = Prediction::latest()->first();
        return json_encode([
            "status" => true,
            "message" => "Success",
            "data" => $data->toArray()
        ]);
    }
}
