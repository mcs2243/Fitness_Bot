import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Upload, Activity, Zap, Moon, FileText, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface DashboardProps {
    data: any[];
    onUpload: (file: File) => Promise<void>;
}

interface UploadedFile {
    name: string;
    status: 'uploading' | 'processing' | 'success' | 'error';
    type?: string;
}

const Dashboard: React.FC<DashboardProps> = ({ data, onUpload }) => {
    const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
    const [isProcessing, setIsProcessing] = useState(false);

    const onDrop = useCallback(async (acceptedFiles: File[]) => {
        if (acceptedFiles.length > 0) {
            const file = acceptedFiles[0];
            const newFile: UploadedFile = { name: file.name, status: 'uploading' };
            setUploadedFiles(prev => [newFile, ...prev]);
            setIsProcessing(true);

            try {
                // Simulate processing delay for better UX if needed, or just await upload
                await onUpload(file);

                setUploadedFiles(prev => prev.map(f =>
                    f.name === file.name ? { ...f, status: 'success', type: 'CSV' } : f
                ));
            } catch (error) {
                setUploadedFiles(prev => prev.map(f =>
                    f.name === file.name ? { ...f, status: 'error' } : f
                ));
            } finally {
                setIsProcessing(false);
            }
        }
    }, [onUpload]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'text/csv': ['.csv'], 'application/json': ['.json'] },
        disabled: isProcessing
    });

    // Process data for charts
    const chartData = data.map(d => ({
        date: new Date(d.date).toLocaleDateString(),
        volume: (d.weight_lb || 0) * (d.reps || 0) * (d.sets || 1),
        recovery: d.recovery_score || 0,
        strain: d.day_strain || 0,
        sleep: d.sleep_performance || 0,
    })).reverse();

    return (
        <div className="p-6 space-y-8 max-w-7xl mx-auto">
            <header className="mb-8">
                <h1 className="text-3xl font-bold text-white mb-2">Fitness Dashboard</h1>
                <p className="text-gray-400">Track your progress and recovery</p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Left Column: Upload & Files */}
                <div className="lg:col-span-1 space-y-6">
                    {/* Upload Section */}
                    <div
                        {...getRootProps()}
                        className={`p-8 border-2 border-dashed rounded-xl text-center cursor-pointer transition-all duration-200
              ${isDragActive ? 'border-blue-500 bg-blue-500/10 scale-[1.02]' : 'border-gray-700 hover:border-gray-500 hover:bg-gray-800/50'}
              ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}
            `}
                    >
                        <input {...getInputProps()} />
                        <div className="relative">
                            {isProcessing ? (
                                <Loader2 className="w-12 h-12 mx-auto text-blue-500 animate-spin mb-4" />
                            ) : (
                                <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                            )}
                        </div>
                        <p className="text-lg font-medium text-gray-300">
                            {isDragActive ? "Drop it like it's hot!" : "Upload Workout Data"}
                        </p>
                        <p className="text-sm text-gray-500 mt-2">
                            Support for <strong>Strong App</strong> & <strong>Whoop</strong> CSVs
                        </p>
                    </div>

                    {/* Uploaded Files List */}
                    <div className="bg-gray-800/50 rounded-xl border border-gray-700 overflow-hidden">
                        <div className="p-4 border-b border-gray-700 bg-gray-800">
                            <h3 className="font-semibold text-gray-200 flex items-center gap-2">
                                <FileText size={18} />
                                Recent Uploads
                            </h3>
                        </div>
                        <div className="max-h-60 overflow-y-auto p-2 space-y-2">
                            <AnimatePresence>
                                {uploadedFiles.length === 0 && (
                                    <p className="text-center text-gray-500 py-4 text-sm">No files uploaded yet.</p>
                                )}
                                {uploadedFiles.map((file, idx) => (
                                    <motion.div
                                        key={`${file.name}-${idx}`}
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        className="flex items-center justify-between p-3 rounded-lg bg-gray-900/50 border border-gray-800"
                                    >
                                        <div className="flex items-center gap-3 overflow-hidden">
                                            <FileText size={16} className="text-gray-400 flex-shrink-0" />
                                            <span className="text-sm text-gray-300 truncate">{file.name}</span>
                                        </div>
                                        <div className="flex-shrink-0">
                                            {file.status === 'uploading' && <Loader2 size={16} className="text-blue-400 animate-spin" />}
                                            {file.status === 'success' && <CheckCircle size={16} className="text-green-400" />}
                                            {file.status === 'error' && <AlertCircle size={16} className="text-red-400" />}
                                        </div>
                                    </motion.div>
                                ))}
                            </AnimatePresence>
                        </div>
                    </div>

                    {/* Status Card */}
                    <div className="bg-blue-900/20 border border-blue-800 p-4 rounded-xl">
                        <h4 className="text-blue-400 font-medium mb-1 flex items-center gap-2">
                            <BotIcon size={16} />
                            AI Coach Status
                        </h4>
                        <p className="text-sm text-blue-200/70">
                            {uploadedFiles.some(f => f.status === 'success')
                                ? "Data loaded. I'm ready to analyze your training!"
                                : "Waiting for data to provide insights."}
                        </p>
                    </div>
                </div>

                {/* Right Column: Stats & Charts */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Stats Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <StatCard
                            icon={<Activity />}
                            label="Volume Load"
                            value={chartData.length > 0 ? chartData[chartData.length - 1].volume.toLocaleString() : "0"}
                            sub="lbs (last session)"
                            color="blue"
                        />
                        <StatCard
                            icon={<Zap />}
                            label="Recovery"
                            value={chartData.length > 0 ? `${chartData[chartData.length - 1].recovery}%` : "-"}
                            sub="Whoop Score"
                            color="green"
                        />
                        <StatCard
                            icon={<Moon />}
                            label="Sleep Perf"
                            value={chartData.length > 0 ? `${chartData[chartData.length - 1].sleep}%` : "-"}
                            sub="Efficiency"
                            color="purple"
                        />
                    </div>

                    {/* Charts */}
                    <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-xl">
                        <h3 className="text-xl font-semibold text-gray-200 mb-6">Volume Progression</h3>
                        <div className="h-72">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                                    <XAxis dataKey="date" stroke="#9CA3AF" tick={{ fontSize: 12 }} tickMargin={10} />
                                    <YAxis stroke="#9CA3AF" tick={{ fontSize: 12 }} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', color: '#fff', borderRadius: '8px' }}
                                        itemStyle={{ color: '#E5E7EB' }}
                                    />
                                    <Line type="monotone" dataKey="volume" stroke="#3B82F6" strokeWidth={3} dot={{ r: 4, fill: '#3B82F6' }} activeDot={{ r: 6 }} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-xl">
                        <h3 className="text-xl font-semibold text-gray-200 mb-6">Recovery vs Strain</h3>
                        <div className="h-72">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                                    <XAxis dataKey="date" stroke="#9CA3AF" tick={{ fontSize: 12 }} tickMargin={10} />
                                    <YAxis stroke="#9CA3AF" tick={{ fontSize: 12 }} />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', color: '#fff', borderRadius: '8px' }}
                                        cursor={{ fill: '#374151', opacity: 0.4 }}
                                    />
                                    <Bar dataKey="recovery" fill="#10B981" radius={[4, 4, 0, 0]} maxBarSize={50} />
                                    <Bar dataKey="strain" fill="#EF4444" radius={[4, 4, 0, 0]} maxBarSize={50} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

const StatCard = ({ icon, label, value, sub, color }: any) => {
    const colors: any = {
        blue: "bg-blue-500/20 text-blue-400",
        green: "bg-green-500/20 text-green-400",
        purple: "bg-purple-500/20 text-purple-400",
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gray-800 p-5 rounded-xl border border-gray-700"
        >
            <div className="flex items-center gap-3 mb-3">
                <div className={`p-2 rounded-lg ${colors[color]}`}>{icon}</div>
                <h3 className="text-sm font-medium text-gray-400">{label}</h3>
            </div>
            <div>
                <p className="text-2xl font-bold text-white">{value}</p>
                <span className="text-xs text-gray-500">{sub}</span>
            </div>
        </motion.div>
    );
};

// Simple Bot Icon component since lucide-react might not export 'Bot' as 'BotIcon' sometimes
const BotIcon = ({ size }: { size: number }) => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width={size}
        height={size}
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
    >
        <path d="M12 8V4H8" />
        <rect width="16" height="12" x="4" y="8" rx="2" />
        <path d="M2 14h2" />
        <path d="M20 14h2" />
        <path d="M15 13v2" />
        <path d="M9 13v2" />
    </svg>
);

export default Dashboard;
