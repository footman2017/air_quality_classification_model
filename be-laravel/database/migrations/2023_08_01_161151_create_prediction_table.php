<?php

use Illuminate\Support\Facades\Schema;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('prediction', function (Blueprint $table) {
            $table->id();
            $table->timestamps();
            $table->unsignedSmallInteger('pm10');
            $table->unsignedSmallInteger('pm25');
            $table->unsignedSmallInteger('so2');
            $table->unsignedSmallInteger('co');
            $table->unsignedSmallInteger('o3');
            $table->unsignedSmallInteger('no2');
            $table->unsignedTinyInteger('prediction_result');
            $table->unsignedTinyInteger('accuracy');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('prediction');
    }
};
